import random

import torch
from torch import nn
import numpy as np


class BEiTv1(nn.Module):
    def __init__(self, model, tokenizer, ratio=0.4, patch_size=16):
        super(BEiTv1, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.patch_size = patch_size

        for param in self.tokenizer.parameters():
            param.requires_grad = False

        fc_dim = self.model.fc.weight.shape[1]
        vocab_dim = self.tokenizer.out_dim
        dim = 768
        self.fc = nn.Linear(fc_dim, vocab_dim)
        self.mask_token = nn.Parameter(torch.rand(1, 1, dim))

    def get_mask(self, h, w):
        n = h * w // self.patch_size ** 2
        min_num = int(n * self.ratio)
        index = np.arange(n).reshape(h // self.patch_size, w // self.patch_size)

        mask = []
        while len(mask) < min_num:
            size = random.randint(16, min_num)
            ratio = random.uniform(0.3, 1/0.3)
            h_len = int((size / ratio) ** 0.5)
            w_len = int((size * ratio) ** 0.5)
            h_start = random.randint(0, h - h_len)
            w_start = random.randint(0, w - w_len)
            new_square_mask = index[h_start:h_start+h_len, w_start:w_start+w_len]
            mask = np.union1d(mask, new_square_mask)

        return mask.reshape(1, -1)

    def forward(self, x):
        visual_token = self.tokenizer(x)
        mask = self.get_mask(*x.shape[2:])
        x = self.model.patch_embed(x)
        x[mask] = self.mask_token
        # do something
        logit = self.fc(x[mask])
        label = visual_token[mask]

        return logit, label