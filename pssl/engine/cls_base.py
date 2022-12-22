import datetime
import time

import torch

from pic.utils import Metric, Accuracy


@torch.inference_mode()
def test(valid_dataloader, model, args):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    top1_m = Metric(reduce_every_n_step=args.print_freq, header='Top-1:')
    top5_m = Metric(reduce_every_n_step=args.print_freq, header='Top-5:')

    # 2. start validate
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(valid_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(valid_dataloader):
        batch_size = x.size(0)

        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            y_hat = model(x)

        top1, top5 = Accuracy(y_hat, y, top_k=(1,5,))

        top1_m.update(top1, batch_size)
        top5_m.update(top5, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"TEST: [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {top1_m} {top5_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    top1 = top1_m.compute()
    top5 = top5_m.compute()

    # 4. print metric
    space = 16
    num_metric = 6
    args.log('-'*space*num_metric)
    args.log(("{:>16}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Top-1 Acc', 'Top-5 Acc'))
    args.log('-'*space*num_metric)
    args.log(f"{'TEST':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{top1:{space}.4f}{top5:{space}.4f}")
    args.log('-'*space*num_metric)

    return top1, top5
