from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import os
import net

def _forward_task(model, x):

    x = getattr(model, 'classifier_conv')(x)
    x = getattr(model, 'classifier_maxpool')(x)
    x = x.squeeze()
    x = nn.Sigmoid()(x)

    return x

def load_model(N_LABELS, opt):

    if opt.arch.startswith("resnet"):

        model = net.resnet(opt.arch, pretrained=True, num_classes=1000,
                           zero_init_residual=True, module=opt.module, opt=opt)
        print(model)
        num_ftrs = model.fc.in_features

        # remove unnecessary modules to save gpu mem
        modules_rm = ['fc', "avgpool"]

        for module_rm in modules_rm:
            delattr(model, module_rm)

        setattr(model, 'classifier_conv', nn.Conv2d(num_ftrs, N_LABELS, kernel_size=1))
        setattr(model, 'classifier_maxpool', nn.AdaptiveMaxPool2d((1, 1)))
        setattr(model, '_forward_task', _forward_task)

    else:
        raise ValueError('wrong type of architecture', opt.arch)

    # add final layer with # outputs in same dimension of labels with sigmoid activation
    # put model on GPU
    model = model.cuda(opt.gpu_ids[0])
    model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model


def checkpoint(model, best_loss, epoch, LR, opt):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state_dict = model.state_dict()
    state = {
        'state_dict': state_dict,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, os.path.join(opt.run_path, 'checkpoint'))


def create_optimizer(model, LR, WEIGHT_DECAY, opt):
    params = [
        {'params': [p for p in model.parameters()]},
    ]

    optimizer = optim.SGD(
        params,
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)

    return optimizer
