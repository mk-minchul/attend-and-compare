from __future__ import print_function, division
import torch
import os
import time
import numpy as np
import csv
import eval_model as E
import wandb
from model import load_model, checkpoint

def train_model(
        model,
        criterion,
        optimizer,
        LR,
        scheduler,
        num_epochs,
        dataloaders,
        dataset_sizes,
        PATH_TO_IMAGES,
        data_transforms,
        opt,
):
    """
    Fine tunes torchvision model to NIH CXR data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_auc = -1
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}(max)'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or val
        # necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            print('Epoch %03d, ' % epoch, phase)
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            data_length = len(dataloaders[phase])

            for data_idx, data in enumerate(dataloaders[phase]):
                inputs, labels, _ = data
                batch_size = inputs.shape[0]

                if phase == 'val':
                    with torch.no_grad():
                        inputs = inputs.cuda(opt.gpu_ids[0])
                        labels = labels.cuda(opt.gpu_ids[0]).float()
                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            # has dot product
                            outputs, dp = outputs
                        else:
                            dp = None

                        # calculate gradient and update parameters in train phase
                        optimizer.zero_grad()

                        loss = criterion(outputs, labels)
                else:
                    inputs = inputs.cuda(opt.gpu_ids[0])
                    labels = labels.cuda(opt.gpu_ids[0]).float()
                    outputs = model(inputs)

                    if isinstance(outputs, tuple):
                        # has dot product
                        outputs, dp = outputs
                    else:
                        dp = None

                    # calculate gradient and update parameters in train phase
                    optimizer.zero_grad()

                    loss = criterion(outputs, labels)

                    if dp is not None:
                        dp_loss = opt.orth_loss_lambda * torch.abs(dp.mean())
                        loss = loss + dp_loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    if data_idx % 20 == 0:
                        wandb.log({
                            'epoch': epoch + data_idx / float(len(dataloaders[phase])),
                            'loss': loss.cpu(),
                            'lr': list(optimizer.param_groups)[0]['lr']
                        })

                if data_idx == 0:
                    log_images = []
                    for image in list(inputs[:10].cpu()):
                        log_images.append(wandb.Image(
                            np.transpose(image.numpy(), (1, 2, 0)),
                            caption='{}_image'.format(phase)
                        ))

                    wandb.log({'{}_image'.format(phase): log_images})

                running_loss += loss.data.item() * batch_size

                if data_idx % 100 == 0:
                    print("{} / {} ".format(data_idx, data_length), end="\r", flush=True)

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch
            if phase == 'val':
                pred, auc = E.make_pred_multilabel(
                    data_transforms,
                    model,
                    PATH_TO_IMAGES,
                    fold="val",
                    opt=opt,
                )
                wandb.log({
                    'epoch': epoch + 1,
                    'performance': np.average(list(auc.auc))
                })

                epoch_auc = np.average(list(auc.auc))
                scheduler.step(epoch_auc)

            # checkpoint model
            if phase == 'val' and epoch_auc > best_auc:
                # best_loss = epoch_loss
                best_auc = epoch_auc
                best_epoch = epoch

                checkpoint(model, best_auc, epoch, LR, opt)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open(os.path.join(opt.run_path, "log_train"), 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if (epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if (total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if np.round(list(optimizer.param_groups)[0]['lr'], 5) <= np.round(
                LR * (opt.lr_decay_ratio ** opt.num_lr_drops), 5):
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load(os.path.join(opt.run_path, 'checkpoint'))
    model = load_model(N_LABELS=14, opt=opt)
    model.load_state_dict(checkpoint_best['state_dict'])

    return model, best_epoch
