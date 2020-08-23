from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np

import cxr_dataset as CXR
import eval_model as E
from utils import set_seed
import wandb
from model import load_model, create_optimizer
from train import train_model

def run(PATH_TO_IMAGES, LR, WEIGHT_DECAY, opt):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        PATH_TO_IMAGES: path to NIH images
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD

    Returns:
        preds: torchvision model predictions on test fold with ground truth for comparison
        aucs: AUCs for each train,test tuple

    """

    use_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    print("Available GPU count:" + str(gpu_count))

    wandb.init(project=opt.project, name=opt.run_name)
    wandb.config.update(opt, allow_val_change=True)

    NUM_EPOCHS = 60
    BATCH_SIZE = opt.batch_size

    if opt.eval_only:
        # test only. it is okay to have duplicate run_path
        os.makedirs(opt.run_path, exist_ok=True)
    else:
        # train from scratch, should not have the same run_path. Otherwise it will overwrite previous runs.
        try:
            os.makedirs(opt.run_path)
        except FileExistsError:
            print("[ERROR] run_path {} exists. try to assign a unique run_path".format(opt.run_path))
            return None, None
        except Exception as e:
            print("exception while creating run_path {}".format(opt.run_path))
            print(str(e))
            return None, None

    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    N_LABELS = 14  # we are predicting 14 labels

    # define torchvision transforms
    if opt.random_crop:

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(size=opt.input_size, scale=(0.8, 1.0)),  # crop then resize
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(int(opt.input_size * 1.05)),
                transforms.CenterCrop(opt.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }

    else:
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(opt.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(opt.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        }
    # create train/val dataloaders
    transformed_datasets = {}
    transformed_datasets['train'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'])
    transformed_datasets['val'] = CXR.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'])

    worker_init_fn = set_seed(opt)

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=30,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=30,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")

    # load model
    model = load_model(N_LABELS, opt)

    # define criterion, optimizer for training
    criterion = nn.BCELoss()

    optimizer = create_optimizer(model, LR, WEIGHT_DECAY, opt)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'max',
        factor=opt.lr_decay_ratio,
        patience=opt.patience,
        verbose=True
    )

    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'val']}

    if opt.eval_only:
        print("loading best model statedict")
        # load best model weights to return
        checkpoint_best = torch.load(os.path.join(opt.run_path, 'checkpoint'))
        model = load_model(N_LABELS, opt=opt)
        model.load_state_dict(checkpoint_best['state_dict'])

    else:
        # train model
        model, best_epoch = train_model(
            model,
            criterion,
            optimizer,
            LR,
            scheduler=scheduler,
            num_epochs=NUM_EPOCHS,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            PATH_TO_IMAGES=PATH_TO_IMAGES,
            data_transforms=data_transforms,
            opt=opt,
        )

    # get preds and AUCs on test fold
    preds, aucs = E.make_pred_multilabel(
        data_transforms,
        model,
        PATH_TO_IMAGES,
        fold="test",
        opt=opt,
    )

    wandb.log({
        'val_official': np.average(list(aucs.auc))
    })

    return preds, aucs

