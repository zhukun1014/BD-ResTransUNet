# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import math


class SynergisticOperator(nn.Module):
    """Abstract Base Class for Dual-Stream Operators."""

    def __init__(self):
        super(SynergisticOperator, self).__init__()


class GeometricAdaptiveHDA(SynergisticOperator):
    """
    Innovation 1: Geometric-Adaptive Hybrid Deformable Attention (HDA).
    Addresses non-rigid SLE lesion topologies via dynamic receptive field reconfiguration.
    """

    def __init__(self, dim, out_dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=8):
        super().__init__()
        self.dim = dim
        self.groups = groups
        self.kernel_size = kernel_size

        # Offset and Mask Generation for Deformable Sampling
        channels_ = groups * 3 * kernel_size * kernel_size
        self.offset_generator = nn.Conv2d(dim, 2 * kernel_size ** 2 * groups, kernel_size, stride, padding)
        self.mask_generator = nn.Conv2d(dim, kernel_size ** 2 * groups, kernel_size, stride, padding)

        # Weight parameters with Kaiming Initialization
        self.weight = nn.Parameter(torch.empty(out_dim, dim // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_dim))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        offset = self.offset_generator(x)
        mask = torch.sigmoid(self.mask_generator(x))

        return deform_conv2d(x, offset, self.weight, self.bias,
                             stride=(1, 1), padding=(1, 1),
                             dilation=(1, 1), mask=mask)


class MultiScaleSynergisticFusion(SynergisticOperator):
    """
    Innovation 2: MSF Gateway.
    Synchronizes heterogeneous feature maps via cross-resolution semantic alignment.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, out_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, out_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, out_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, semantic_map, structural_map):
        B, C, H, W = semantic_map.size()
        # Non-local interaction logic for multi-scale fusion
        proj_query = self.query_conv(semantic_map).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key_conv(structural_map).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.value_conv(structural_map).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + semantic_map