import os
from tqdm import tqdm
import pandas as pd

import torch
from torch import distributed as dist
from torch.nn import functional as F

from pssl.utils import all_reduce_sum, Metric, knn_classifier, Accuracy
from pssl.utils.metric import all_gather_with_different_size


@torch.inference_mode()
def extract_features(dataloader, model, args, split):
    dataset_len = len(dataloader.dataset)
    data_type = torch.half if args.amp else torch.float
    whole_features = torch.zeros([dataset_len, args.feat_dim], device=args.device, dtype=data_type)
    whole_labels = torch.zeros([dataset_len], device=args.device, dtype=torch.long)
    is_single = os.environ.get('LOCAL_RANK', None) is None
    is_master = is_single or int(os.environ['LOCAL_RANK']) == 0

    model.eval()
    for (x, y), idx in tqdm(dataloader, desc='extract features', disable=not is_master):
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        with torch.cuda.amp.autocast(args.amp):
            features = model(x)
            whole_features[idx] = features
            whole_labels[idx] = y

    if args.distributed:
        whole_features = all_reduce_sum(whole_features)
        whole_labels = all_reduce_sum(whole_labels)

    if is_master:
        whole_features = F.normalize(whole_features, dim=-1)
        torch.save(whole_features.cpu(), os.path.join(args.log_dir, f"{split}_features.pth"))
        torch.save(whole_labels.cpu(), os.path.join(args.log_dir, f"{split}_labels.pth"))
        args.log(f"{split} features and labels are saved into {args.log_dir}")

    dist.barrier()



def run_test(valid_feature, valid_label, train_feature, train_label, k, t, num_classes):
    top1_m = Metric(reduce_every_n_step=None, reduce_on_compute=False, header='Top-1:')
    top5_m = Metric(reduce_every_n_step=None, reduce_on_compute=False, header='Top-5:')

    is_single = os.environ.get('LOCAL_RANK', None) is None
    is_master = is_single or int(os.environ['LOCAL_RANK']) == 0
    prog_bar = tqdm(zip(valid_feature, valid_label), total=len(valid_feature), disable=not is_master)

    for x, y in prog_bar:
        batch_size = x.size(0)
        y_hat = knn_classifier(x, train_feature, train_label, num_classes, k, t)

        top1, top5 = Accuracy(y_hat, y, top_k=(1, 5,))
        top1_m.update(top1, batch_size)
        top5_m.update(top5, batch_size)

        prog_bar.set_description(f"KNN (k={k}, t={t}) {top1_m} {top5_m}")

    top1 = top1_m.compute()
    top5 = top5_m.compute()

    return top1, top5


def test_with_knn_classifier(train_dataloader, test_dataloader, model, args):
    # 1. save features if not saved before
    if args.feature_path is None:
        extract_features(train_dataloader, model, args, 'train')
        extract_features(test_dataloader, model, args, 'val')
        args.feature_path = args.log_dir

    # 2. load features
    args.log(f"load saved features from {args.feature_path}")
    train_feature = torch.load(os.path.join(args.feature_path, f'train_features.pth')).to(args.device)
    train_label = torch.load(os.path.join(args.feature_path, f'train_labels.pth')).to(args.device)
    valid_feature = torch.load(os.path.join(args.feature_path, f'val_features.pth')).to(args.device)
    valid_label = torch.load(os.path.join(args.feature_path, f'val_labels.pth')).to(args.device)

    chunk_size = 256
    train_feature = train_feature.t()
    valid_feature = valid_feature.split(chunk_size, dim=0)
    valid_label = valid_label.split(chunk_size, dim=0)

    # 3. run evaluation
    result = list()
    t_k_list = torch.tensor([[t, k] for t in args.t for k in args.k]).tensor_split(args.world_size)[args.gpu]
    for x in t_k_list:
        t, k = round(float(x[0].item()), 3), int(x[1].item())
        x = x.to(args.device)
        top1, top5 = run_test(valid_feature, valid_label, train_feature, train_label, k, t, args.num_classes)
        result.append(torch.cat([x, top1.view(1), top5.view(1)], dim=0))

    # 4. (optional) gather result
    result = torch.stack(result, dim=0)
    if args.distributed:
        result = all_gather_with_different_size(result)

    # 5. compute best top1 accuracy
    best_idx = result[:, 2].argmax()
    best_t, best_k, best_top1, best_top5 = result[best_idx].cpu().tolist()

    # 6. display & save knn experiment
    if args.is_rank_zero:
        result = result.cpu().tolist()
        df = pd.DataFrame(result, columns=['t', 'k', 'top1', 'top5']).sort_values(by=['t', 'k'], ignore_index=True)
        df.to_csv(os.path.join(args.log_dir, 'knn_val_result.csv'))
        args.log(f'[Best KNN result] k: {int(best_k)} t: {best_t:0.2f} top1: {best_top1:.03f}% top5: {best_top5:.03f}%')
        args.log(f'validation result on {args.setting} (saved to: {args.log_dir})\n' + df.to_string())

    return best_top1, best_top5

