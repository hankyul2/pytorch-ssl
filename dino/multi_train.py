import os
import sys
import argparse

from pathlib import Path
from traceback import print_exc

from main import run
from pic.utils import get_args_parser, clear


setting_dict = dict(
    dino = "ImageNet "
           "--dataset_type ImageFolder "
           "--drop-path-rate 0.1 "
           "--smoothing 0.0 "
           "--epoch 300 "
           "--optimizer adamw "
           "--weight-decay 1e-4 "
           "--lr 0.0005 "
           "--min-lr 1e-6 "
           "--warmup-lr 0 "
           "--warmup-epoch 10 "
           "--scheduler cosine "
           "--grad-norm 3.0 "
           "--cutmix 0.0 "
           "--mixup 0.0 "
           "--remode 0.0 "
           "-b 64 "
           "-j 8 "
           "--pin-memory "
           "--amp "
           "--drop-last "
           "--channels-last",
)


def get_multi_args_parser():
    parser = argparse.ArgumentParser(description='pytorch-image-classification', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('setup', type=str, nargs='+', choices=setting_dict.keys(), help='experiment setup')
    parser.add_argument('-m', '--model-name', type=str, nargs='+', default=['vit_small'], help='list of model names')
    parser.add_argument('-t', '--model-type', type=str, default='pic', help='model type')
    parser.add_argument('-c', '--cuda', type=str, default='0', help='cuda device')
    parser.add_argument('-o', '--output-dir', type=str, default='log', help='log dir')
    parser.add_argument('-p', '--project-name', type=str, default='ssl', help='project name used for wandb logger')
    parser.add_argument('-w', '--who', type=str, default='Hankyul', help='enter your name')
    parser.add_argument('--use-wandb', action='store_true', default=False, help='use wandb')

    return parser


def pass_required_variable_from_previous_args(args, prev_args=None):
    if prev_args:
        required_vars = ['gpu', 'world_size', 'distributed', 'is_rank_zero', 'device']
        for var in required_vars:
            exec(f"args.{var} = prev_args.{var}")


def save_arguments(args, is_master):
    if is_master:
        print("Multiple Train Setting")
        print(f" - model (num={len(args.model_name)}): {', '.join(args.model_name)}")
        print(f" - setting (num={len(args.setup)}): {', '.join(args.setup)}")
        print(f" - cuda: {args.cuda}")
        print(f" - output dir: {args.output_dir}")

        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(args.output_dir, 'last_multi_args.txt'), 'wt') as f:
            f.write(" ".join(sys.argv))


if __name__ == '__main__':
    is_master = os.environ.get('LOCAL_RANK', None) is None or int(os.environ['LOCAL_RANK']) == 0
    multi_args_parser = get_multi_args_parser()
    multi_args = multi_args_parser.parse_args()
    save_arguments(multi_args, is_master)
    prev_args = None

    for setup in multi_args.setup:
        args_parser = get_args_parser()
        args = args_parser.parse_args(setting_dict[setup].split(' '))
        pass_required_variable_from_previous_args(args, prev_args)
        for model_name in multi_args.model_name:
            # args.epoch=1
            args.setup = setup
            args.exp_name = f"{model_name}_{setup}"
            args.model_name = model_name
            for option_name in ['model_type', 'cuda', 'output_dir', 'project_name', 'who', 'use_wandb']:
                exec(f"args.{option_name} = multi_args.{option_name}")
            try:
                run(args)
            except:
                print(print_exc())
            clear(args)
        prev_args = args