import argparse
import torch
import numpy as np
import random


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(opt):
    torch.cuda.set_device(opt.gpu_ids[0])
    if opt.fix_randomness:
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)
        # documentation says we need this.(computation performance may drop)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # reference: https://github.com/pytorch/pytorch/issues/13555
        def worker_init_fn(worker_id):
            seed = opt.seed + worker_id
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            return
    else:
        torch.backends.cudnn.benchmark = True
        worker_init_fn = None

    return worker_init_fn