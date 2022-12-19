import gc
import os
import argparse

import torch

# from mecla.dataset import get_dataset
# from mecla.model import get_model
# from mecla.engine import test
from pssl.utils import setup, get_args_with_setting, print_batch_run_settings, clear, load_model_list_from_config


def get_args_parser():
    parser = argparse.ArgumentParser(
        description='pytorch-self-supervised-learning (SSL)',
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 1.setup
    setup = parser.add_argument_group('setup')
    setup.add_argument(
        '--config', type=str, default=os.path.join('config', 'valid.json'),
        help='paths for each dataset and pretrained-weight. (json)'
    )
    setup.add_argument(
        '-s', '--settings', type=str, default=['imagenet1k_224_v1'], nargs='+',
        help='settings used for default value'
    )
    setup.add_argument(
        '--eval-protocol', type=str, default='fc', choices=['fc', 'knn'],
        help="choose between fully-connected classifier and knn classifier"
    )
    setup.add_argument(
        '--entity', type=str, default='hankyul2',
        help='project space used for wandb logger'
    )
    setup.add_argument(
        '-proj', '--project-name', type=str, default='pytorch-ssl-valid',
        help='project name used for wandb logger'
    )
    setup.add_argument(
        '--who', type=str, default='hankyul2',
        help='enter your name'
    )
    setup.add_argument(
        '--use-wandb', action='store_true', default=False,
        help='track std out and log metric in wandb'
    )
    setup.add_argument(
        '-exp', '--exp-name', type=str, default=None,
        help='experiment name for each run'
    )
    setup.add_argument(
        '--exp-target', type=str, default=['setting', 'model_name'], nargs='+',
        help='experiment name based on arguments'
    )
    setup.add_argument(
        '-out', '--output-dir', type=str, default='log_val',
        help='where log output is saved'
    )
    setup.add_argument(
        '-p', '--print-freq', type=int, default=50,
        help='how often print metric in iter'
    )
    setup.add_argument(
        '--seed', type=int, default=42,
        help='fix seed'
    )
    setup.add_argument(
        '--amp', action='store_true', default=False,
        help='enable native amp(fp16) training'
    )
    setup.add_argument(
        '--channels-last', action='store_true',
        help='change memory format to channels last'
    )
    setup.add_argument(
        '-c', '--cuda', type=str, default='0,1,2,3,4,5,6,7,8',
        help='CUDA_VISIBLE_DEVICES options'
    )
    setup.set_defaults(amp=True, channel_last=True, pin_memory=True, resume=None, mode='valid')

    # 2. augmentation & dataset & dataloader
    data = parser.add_argument_group('data')
    data.add_argument(
        '--dataset-type', type=str, default='imagenet1k',
        choices=['imagenet1k', 'cifar10', 'cifar100', 'celeba'],
        help='dataset type'
    )
    data.add_argument(
        '--test-size', type=int, default=(224, 224), nargs='+',
        help='test image size'
    )
    data.add_argument(
        '--test-resize-mode', type=str, default='resize_shorter', choices=['resize_shorter', 'resize'],
        help='test resize mode'
    )
    data.add_argument(
        '--center-crop-ptr', type=float, default=0.875,
        help='test image crop percent'
    )
    data.add_argument(
        '--interpolation', type=str, default='bicubic',
        help='image interpolation mode'
    )
    data.add_argument(
        '--mean', type=float, default=(0.485, 0.456, 0.406), nargs='+',
        help='image mean'
    )
    data.add_argument(
        '--std', type=float, default=(0.229, 0.224, 0.225), nargs='+',
        help='image std'
    )
    data.add_argument(
        '-b', '--batch-size', type=int, default=256,
        help='batch size'
    )
    data.add_argument(
        '-j', '--num-workers', type=int, default=8,
        help='number of workers'
    )
    data.add_argument(
        '--pin-memory', action='store_true', default=False,
        help='pin memory in dataloader'
    )

    # 3.model
    model = parser.add_argument_group('model')
    model.add_argument(
        '-m', '--model-names', type=str, default=[], nargs='+',
        help='model name'
    )
    model.add_argument(
        '--model-type', type=str, default='timm',
        help='timm or torchvision'
    )
    model.add_argument(
        '--in-channels', type=int, default=3,
        help='input channel dimension'
    )
    model.add_argument(
        '--drop-path-rate', type=float, default=0.0,
        help='stochastic depth rate'
    )
    model.add_argument(
        '--sync-bn', action='store_true', default=False,
        help='apply sync batchnorm'
    )
    model.add_argument(
        '--pretrained', action='store_true', default=False,
        help='load pretrained weight'
    )

    return parser


def run(args, train_dataloader, test_dataloader):
    # model = get_model(args)
    # result = test(valid_dataloader=valid_dataloader, valid_dataset=valid_dataset, model=model, args=args)
    pass


if __name__ == '__main__':
    # 1. parse command
    parser = get_args_parser()
    args = parser.parse_args()

    # 2. run N(setting) x M(model_names) experiment
    prev_args = None
    for setting in args.settings:
        # 2-1. load model names and print
        args.setting = setting
        if len(args.model_names) == 0:
            args.model_names = load_model_list_from_config(args, args.mode)
        print_batch_run_settings(args)

        # 2-2. load dataset & dataloader
        # train_dataset, test_dataset = get_dataset(new_args, new_args.mode)
        # train_dataloader, test_dataloader = get_dataset(new_args, new_args.mode)
        train_dataset = test_dataset = train_dataloader = test_dataloader = None


        # 2-3. valid each model
        for model_name in args.model_names:
            new_args = get_args_with_setting(parser, args.config, setting, model_name, prev_args, args.mode)
            setup(new_args)

            run(new_args, train_dataloader, test_dataloader)

            clear(new_args)
            prev_args = new_args