import timm
import torch
import torch.nn as nn
import torchvision
from torch.nn.parallel import DistributedDataParallel


def get_model(args):
    tv_model = torchvision.models.__dict__.get(args.model_name)
    if args.model_type == 'torchvision' and tv_model:
        model = tv_model(num_classes=args.num_classes, pretrained=args.pretrained).cuda(args.device)
    elif args.model_type == 'timm' or not tv_model:
        model = timm.create_model(args.model_name, in_chans=args.in_channels, num_classes=args.num_classes, drop_path_rate=args.drop_path_rate, pretrained=args.pretrained).cuda(args.device)
    else:
        raise Exception(f"{args.model_type} is not supported yet")

    if args.checkpoint_path:
        args.log(f"load model weight from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        for key in ['model', 'state_dict']:
            if key in state_dict:
                state_dict = state_dict[key]

        for fc_name in ['fc', 'fc.0', 'fc.2', 'head', 'classifier', 'adv_classifier']:
            fc_weight = f"{fc_name}.weight"
            fc_bias = f"{fc_name}.bias"
            if fc_weight in state_dict and args.num_classes != state_dict[fc_weight].shape[0]:
                args.log('popping out head')
                state_dict.pop(fc_weight)
                state_dict.pop(fc_bias)
        model.load_state_dict(state_dict, strict=False)

    if args.eval_protocol == 'knn':
        for fc_name in ['fc', 'head', 'classifier', 'adv_classifier']:
            if hasattr(model, fc_name):
                exec(f"args.feat_dim = model.{fc_name}.weight.shape[1]")
                exec(f"model.{fc_name} = nn.Identity()")

    return model

def get_ddp_model(model, args):
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.distributed:
        ddp_model = DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        ddp_model = None

    return model, ddp_model

