import torch
import torch.nn.functional as F
from torch import nn
import random
#import ipdb as pdb


class DropBlock2D(nn.Module):
    r"""Randomly zeroes spatial blocks of the input tensor.


    As described in the paper
    `DropBlock: A regularization method for convolutional networks`_ ,
    dropping whole blocks of feature map allows to remove semantic
    information as compared to regular dropout.

    Args:
        keep_prob (float, optional): probability of an element to be kept.
        Authors recommend to linearly decrease this value from 1 to desired
        value.
        block_size (int, optional): size of the block. Block size in paper
        usually equals last feature map dimensions.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    .. _DropBlock: A regularization method for convolutional networks:
       https://arxiv.org/abs/1810.12890
    """

    def __init__(self, keep_prob=0.9, block_size=7):
        super(DropBlock2D, self).__init__()
        self.keep_prob = keep_prob
        self.block_size = block_size

    def forward(self, input):
        if not self.training or self.keep_prob == 1:
            return input
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        for sh in input.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        M = torch.bernoulli(torch.ones_like(input) * gamma)
        Msum = F.conv2d(M,
                        torch.ones((input.shape[1], 1, self.block_size, self.block_size)).to(device=input.device,
                                                                                             dtype=input.dtype),
                        padding=self.block_size // 2,
                        groups=input.shape[1])
        torch.set_printoptions(threshold=5000)
        mask = (Msum < 1).to(device=input.device, dtype=input.dtype)
        return input * mask * mask.numel() /mask.sum() #TODO input * mask * self.keep_prob ?


class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio
    
    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = int(self.h_ratio * h)
            rw = int(self.w_ratio * w)
            sx = random.randint(0, h-rh)
            sy = random.randint(0, w-rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 0
            x = x * mask
        return x


class SamplePart(nn.Module):
    def __init__(self, part_size, parts_num):
        super(SamplePart, self).__init__()
        self.part_size = int(part_size)
        self.parts_num = int(parts_num)
    
    def forward(self, x, i):
        if self.training:
            h, w = x.size()[-2:]
            rh = int(h / self.part_size)
            rw = w
            sx = int(h / self.parts_num * i)
            sy = 0
            mask = x.new_zeros(x.size())
            mask[:, :, sx:sx+rh, sy:sy+rw] = 1
            return x * mask
        return x
