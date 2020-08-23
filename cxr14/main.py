import sys
import argparse
import run
import random
import string
import os
from utils import str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Options to train the model.')

    parser.add_argument('--path_to_images', type=str, required=True)
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--project', type=str, required=True)

    parser.add_argument('--eval_only', type=str2bool, default='False')
    parser.add_argument('--input_size', type=int, default=336)
    parser.add_argument('--random_crop', type=str2bool, default="False")
    parser.add_argument('--gpu_ids', type=str, default="0")

    # module options
    parser.add_argument('--arch', type=str, default="resnet50")
    parser.add_argument('--module', type=str, default="none")
    parser.add_argument('--num_acm_groups', type=int, default=32)

    parser.add_argument('--fix_randomness', type=str2bool, default='False')
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--run_path', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_lr_drops', type=int, default=2)

    # optimization
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_decay_ratio', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--n_epochs_to_decay', type=str, default='20,10')

    parser.add_argument('--orth_loss_lambda', type=float, default=0.1)

    opt = parser.parse_args(sys.argv[1:])
    opt.n_epochs_to_decay = [int(epoch) for epoch in opt.n_epochs_to_decay.split(',')]
    if "loss" in opt.module:
        opt.diff_loss = True

    if opt.run_path is None:
        opt.run_path = os.path.join(
            'results',
            ''.join(random.choice(string.ascii_letters) for i in range(8))
        )

    opt.gpu_ids = [int(gpu_id) for gpu_id in opt.gpu_ids.split(',')]
    preds, aucs = run.run(opt.path_to_images, opt.lr, opt.weight_decay, opt)
