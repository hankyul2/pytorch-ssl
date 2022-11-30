import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from timm.utils import ModelEmaV2


def all_gather(x):
    if dist.is_initialized():
        dest = [torch.zeros_like(x) for _ in
                range(dist.get_world_size())]
        dist.all_gather(dest, x)
        return torch.cat(dest, dim=0)
    else:
        return x


class MOCO(nn.Module):
    def __init__(self, encoder, dim=128, m=0.999,
                 K=65536, T=0.07):
        super().__init__()
        self.m = m
        self.K = K
        self.T = T

        self.encoder_q = encoder
        self.encoder_k = ModelEmaV2(encoder, decay=m)
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        self.register_buffer("queue",
                             F.normalize(torch.rand(dim, K), dim=1))
        self.register_buffer("queue_ptr",
                             torch.zeros([1], dtype=torch.long))

    def dequeue_enqueue(self, x):
        x_cat = all_gather(x)
        b = x_cat.size(0)
        ptr = int(self.queue_ptr[0])

        self.queue[:, ptr:ptr+b] = x_cat.T
        self.queue_ptr[0] = (ptr + b) % self.K

    def shuffle_batch(self, x):
        x_cat = all_gather(x)
        b_org = x.size(0)
        b = x_cat.size(0)
        rank = dist.get_rank()

        idx2rand = torch.randperm(b).to(x.device)
        dist.broadcast(idx2rand, src=0)
        rand2idx = torch.argsort(idx2rand)

        x = x_cat[idx2rand][b_org*rank:b_org*(rank+1)]

        return x, rand2idx

    def unshuffle_batch(self, x, idx):
        x_cat = all_gather(x)
        b_org = x.size(0)
        rank = dist.get_rank()

        x = x_cat[idx][b_org*rank:b_org*(rank+1)]

        return x

    def forward(self, q, k):
        enc_q = self.encoder_q(q)
        enc_q = F.normalize(enc_q, dim=1)

        with torch.no_grad():
            self.encoder_k.update(self.encoder_q)

            if dist.is_initialized():
                k, idx = self.shuffle_batch(k)

            enc_k = self.encoder_k.module(k)
            enc_k = F.normalize(enc_k, dim=1)

            if dist.is_initialized():
                enc_k = self.unshuffle_batch(enc_k, idx)

        pos = torch.einsum("nc,nc->n",
                           [enc_q, enc_k]).unsqueeze(-1)
        neg = torch.einsum("nc,ck->nk",
                           [enc_q,
                            self.queue.clone().detach()])
        logit = torch.cat([pos, neg], dim=1) / self.T
        label = torch.zeros(q.size(0))\
            .to(torch.long).to(q.device)

        self.dequeue_enqueue(enc_k)

        return logit, label


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]