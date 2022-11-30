import time
import datetime

import timm
import torch
from torchvision import transforms

from pic.utils import setup, get_args_parser, save_checkpoint, resume_from_checkpoint
from pic.utils import print_metadata, Metric, reduce_mean, Accuracy
from pic.data import get_dataset, get_dataloader
from pic.model import get_ema_ddp_model
from pic.criterion import get_scaler_criterion
from pic.optimizer import get_optimizer_and_scheduler

from moco import TwoCropsTransform, MOCO


def train_one_epoch(train_dataloader, model, optimizer, criterion,
                    args, ema_model=None, scheduler=None, scaler=None, epoch=None):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    top1_m = Metric(reduce_every_n_step=args.print_freq, header='Top-1:')
    top5_m = Metric(reduce_every_n_step=args.print_freq, header='Top-5:')
    loss_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Loss:')

    # 2. start validate
    model.train()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(train_dataloader)
    start_time = time.time()

    for batch_idx, ((x1, x2), y) in enumerate(train_dataloader):
        batch_size = x1.size(0)

        if not args.prefetcher:
            x1 = x1.to(args.device)
            x2 = x2.to(args.device)
            y = y.to(args.device)

        if args.channels_last:
            x1 = x1.to(memory_format=torch.channels_last)
            x2 = x2.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat, target = model(x1, x2)
            loss = criterion(y_hat, target)

        top1, top5 = Accuracy(y_hat, target, top_k=(1,5,))

        if args.distributed:
            loss = reduce_mean(loss, args.world_size)

        if args.amp:
            scaler(loss, optimizer, model.parameters(), scheduler, args.grad_norm, batch_idx % args.grad_accum == 0)
        else:
            loss.backward()
            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            if batch_idx % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

        top1_m.update(top1, batch_size)
        top5_m.update(top5, batch_size)
        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} "
                     f"{loss_m} {top1_m} {top5_m}")

        if batch_idx and ema_model and batch_idx % args.ema_update_step == 0:
            ema_model.update(model)

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    top1 = top1_m.compute()
    top5 = top5_m.compute()
    loss = loss_m.compute()

    # 4. print metric
    space = 16
    num_metric = 7
    args.log('-'*space*num_metric)
    args.log(("{:>16}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Loss', 'Top-1 Acc', 'Top-5 Acc'))
    args.log('-'*space*num_metric)
    args.log(f"{'TRAIN('+str(epoch)+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}"
             f"{loss:{space}.4f}{top1:{space}.4f}{top5:{space}.4f}")
    args.log('-'*space*num_metric)

    return loss, top1, top5


def run(args):
    # 0. init ddp & logger
    setup(args)

    # 1. load dataset
    train_dataset, val_dataset = get_dataset(args)
    train_dataset.transform = TwoCropsTransform(transforms.Compose([
        transforms.RandomResizedCrop(args.train_size, scale=args.random_crop_scale),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(args.hflip),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ]))
    train_dataloader, _ = get_dataloader(train_dataset, val_dataset, args)

    # 2. make model
    model = timm.create_model(args.model_name, in_chans=args.in_channels, num_classes=128,
                              drop_path_rate=args.drop_path_rate, pretrained=args.pretrained)
    model = MOCO(model).cuda(args.device)
    model, ema_model, ddp_model = get_ema_ddp_model(model, args)

    # 3. load optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    # 4. load criterion
    criterion, valid_criterion, scaler = get_scaler_criterion(args)

    # 5. print metadata
    print_metadata(model, train_dataset, val_dataset, args)

    # 6. control logic for checkpoint & validate
    if args.resume:
        start_epoch = resume_from_checkpoint(args.checkpoint_path, optimizer, scaler, scheduler)
    else:
        start_epoch = 0

    start_epoch = args.start_epoch if args.start_epoch else start_epoch
    end_epoch = args.end_epoch if args.end_epoch else args.epoch

    if scheduler is not None and start_epoch:
        # Todo: sequential lr does not support step with epoch as positional variable
        scheduler.step(start_epoch)

    # 7. train
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        train_loss, top1, top5 = train_one_epoch(train_dataloader, ddp_model if args.distributed else model, optimizer, criterion, args, ema_model, scheduler, scaler, epoch)

        if args.use_wandb:
            args.log({'train_loss':train_loss, 'top1':top1, 'top5':top5}, metric=True)

        if args.save_checkpoint and args.is_rank_zero:
            save_checkpoint(args.log_dir, model, ema_model, optimizer, scaler, scheduler, epoch)


if __name__ == '__main__':
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    run(args)