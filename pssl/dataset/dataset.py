import os

from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, FashionMNIST

from .transform import TrainTransform, ValTransform


_dataset_dict = {
    'ImageFolder': ImageFolder,
    'CIFAR10': CIFAR10,
    'CIFAR100': CIFAR100,
    'FashionMNIST': FashionMNIST,
}

def register_dataset(fn):
    dataset_name = fn.__name__
    if dataset_name not in _dataset_dict:
        _dataset_dict[fn.__name__] = fn
    else:
        raise ValueError(f"{dataset_name} already exists in dataset_dict")

    return fn


def get_dataset(args, mode):
    dataset_class = _dataset_dict[args.dataset_type]

    if mode == 'train':
        train_transform = TrainTransform(args.train_size, args.train_resize_mode, args.random_crop_pad, args.random_crop_scale,
                                         args.random_crop_ratio, args.hflip, args.auto_aug, args.remode, args.interpolation, args.mean, args.std)
        val_transform = ValTransform(args.test_size, args.test_resize_mode, args.center_crop_ptr, args.interpolation, args.mean, args.std, args.prefetcher)
    else:
        train_transform = val_transform = ValTransform(args.test_size, args.test_resize_mode, args.center_crop_ptr, args.interpolation,
                                     args.mean, args.std)

    if args.dataset_type in ['ImageFolder', 'ImageFolder_with_Idx']:
        train_dataset = dataset_class(os.path.join(args.data_dir, args.train_split), train_transform)
        val_dataset = dataset_class(os.path.join(args.data_dir, args.val_split), val_transform)
        args.num_classes = len(train_dataset.classes)
    elif args.dataset_type in _dataset_dict.keys():
        train_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = dataset_class(root=args.data_dir, train=False, download=True, transform=val_transform)
        args.num_classes = len(train_dataset.classes)
    else:
        assert f"{args.dataset_type} is not supported yet. Just make your own code for it"

    return train_dataset, val_dataset