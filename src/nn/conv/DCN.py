#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
Version:
Autor: dreamy-xay
Date: 2024-10-22 09:49:15
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 09:49:20
"""

import torch
from torch import nn
import math
from torchvision.ops import deform_conv2d
from torch.nn.modules.utils import _pair


class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        output = deform_conv2d(x, offset=offset, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), stride=self.stride, padding=self.padding, dilation=self.dilation, mask=mask)

        return output

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(" f"{self.in_channels}" f", {self.out_channels}" f", kernel_size={self.kernel_size}" f", stride={self.stride}"
        s += f", padding={self.padding}" if self.padding != (0, 0) else ""
        s += f", dilation={self.dilation}" if self.dilation != (1, 1) else ""
        s += f", deformable_groups={self.deformable_groups}" if self.deformable_groups != 1 else ""
        s += ")"

        return s
