import random

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms as TF

from timm.utils import ModelEmaV2


class DINOHead(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048, bottleneck_dim=256, output_dim=65536):
        super(DINOHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        # fix magnitude (weight_g) to 1 (normalization)
        self.fc = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
        self.fc.weight_g.data.fill_(1)
        self.fc.weight_g.requires_grad = False
        self.fc.weight = self.fc.weight_v.detach()

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=1)
        x = self.fc(x)

        return x


class DINO(nn.Module):
    def __init__(self, model, output_dim=65536, m=0.9995, cm=0.9,
                 Ts=0.1, Tt=0.07, Tt_w=0.04, Tt_e=30):
        super().__init__()
        self.m = m
        self.cm = cm
        self.Ts = Ts
        self.Tt = np.concatenate([np.linspace(Tt_w, Tt, Tt_e), np.full(900, Tt)])

        self.student = model
        self.embed_dim = model.head.weight.shape[1]
        self.student.head = DINOHead(self.embed_dim)

        self.teacher = ModelEmaV2(self.student, decay=m)
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.register_buffer("C", torch.rand(1, output_dim))

    def forward(self, xs, epoch=0):
        # 1. forward student
        s = torch.cat([
            self.student(torch.cat(xs[:2], dim=0)),
            self.student(torch.cat(xs[2:], dim=0)),
        ], dim=0)
        s = self.student.head(s).chunk(len(xs))

        # 2. forward teacher
        with torch.no_grad():
            t = self.teacher.module(torch.cat(xs[:2], dim=0))
            t = self.teacher.module.head(t).chunk(2)

        # 3. sharpening+centering
        logit = torch.cat([s[i] / self.Ts for i in [0, 1] + list(range(2, len(xs))) * 2])
        label = torch.cat([(t[i]-self.C) / self.Tt[epoch] for i in [1, 0] + [0] * (len(xs) - 2) + [1] * (len(xs) - 2)])

        # 4. softmax(label) for computing c.e.
        label = F.softmax(label, dim=-1)

        # 5. update center
        t = torch.cat(t)
        dist.all_reduce(t)
        t /= dist.get_world_size()
        self.C[:] = self.C * self.cm + t.mean(dim=0) * (1 - self.cm)

        return logit, label


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    You should this pil based version because hyper-parameters are difficult to convert from pil to torchvision.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class LocalGlobalTransform(object):
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), local_crops_number=8):
        flip_jitter_gray = TF.Compose([
            TF.RandomHorizontalFlip(p=0.5),
            TF.RandomApply([TF.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            TF.RandomGrayscale(p=0.2),
        ])
        normalize = TF.Compose([
            TF.ToTensor(),
            TF.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        interpolation = TF.functional.InterpolationMode('bicubic')

        # global crop 1: GaussianBlur
        self.global1 = TF.Compose([
            TF.RandomResizedCrop(224, scale=global_crops_scale, interpolation=interpolation),
            flip_jitter_gray,
            GaussianBlur(1.0),
            normalize,
        ])
        # global crop 2: GaussianBlur + Solarization
        self.global2 = TF.Compose([
            TF.RandomResizedCrop(224, scale=global_crops_scale, interpolation=interpolation),
            flip_jitter_gray,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # local crop: GaussianBlur
        self.local_crops_number = local_crops_number
        self.local = TF.Compose([
            TF.RandomResizedCrop(96, scale=local_crops_scale, interpolation=interpolation),
            flip_jitter_gray,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = [self.global1(image), self.global2(image)]
        for _ in range(self.local_crops_number):
            crops.append(self.local(image))
        return crops