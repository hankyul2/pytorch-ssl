from math import floor

from timm.data import rand_augment_transform
from torchvision import transforms


class TrainTransform:
    def __init__(self, resize, resize_mode, pad, scale, ratio, hflip, auto_aug, remode, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        transform_list = []

        if hflip:
            transform_list.append(transforms.RandomHorizontalFlip(hflip))

        if auto_aug:
            if auto_aug.startswith('ra'):
                transform_list.append(transforms.RandAugment(interpolation=interpolation))
            elif auto_aug.startswith('ta_wide'):
                transform_list.append(transforms.TrivialAugmentWide(interpolation=interpolation))
            elif auto_aug.startswith('aa'):
                policy = transforms.AutoAugmentPolicy('imagenet')
                transform_list.append(transforms.AutoAugment(policy=policy, interpolation=interpolation))
            elif auto_aug.startswith('timm-ra'):
                transform_list.append(rand_augment_transform('rand-m9-mstd0.5', {}))

        if resize_mode == 'RandomResizedCrop':
            transform_list.append(transforms.RandomResizedCrop(resize, scale=scale, ratio=ratio, interpolation=interpolation))
        elif resize_mode == 'ResizeRandomCrop':
            transform_list.extend([transforms.Resize(resize, interpolation=interpolation),
                                   transforms.RandomCrop(resize, padding=pad)])
        else:
            assert f"{resize_mode} should be RandomResizedCrop and ResizeRandomCrop"

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if remode:
            transform_list.append(transforms.RandomErasing(remode))

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


class ValTransform:
    def __init__(self, size, resize_mode, crop_ptr, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        if not isinstance(size, (tuple, list)):
            size = (size, size)

        resize = (int(floor(size[0] / crop_ptr)), int(floor(size[1] / crop_ptr)))

        if resize_mode == 'resize_shorter':
            resize = resize[0]

        transform_list = [
            transforms.Resize(resize, interpolation=interpolation),
            transforms.CenterCrop(size),
        ]

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)