#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
Version:
Autor: dreamy-xay
Date: 2024-10-22 10:23:28
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 10:24:19
"""
import math
import numpy as np
from torch import nn
from nn.conv.DCN import DCNv2

BN_MOMENTUM = 0.1


class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(nn.BatchNorm2d(cho, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))

        self.conv = DCNv2(chi, cho, kernel_size=3, stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f, padding=f // 2, output_padding=0, groups=o, bias=False)
            self._fill_up_weights(up)

            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
            setattr(self, "node_" + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, "up_" + str(i - startp))
            project = getattr(self, "proj_" + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, "node_" + str(i - startp))
            layers[i] = node(layers[i - 1] + layers[i])

    @staticmethod
    def _fill_up_weights(up):
        w = up.weight.data
        f = math.ceil(w.size(2) / 2)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        for i in range(w.size(2)):
            for j in range(w.size(3)):
                w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, w.size(0)):
            w[c, 0, :, :] = w[0, 0, :, :]


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super(DLAUp, self).__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, "ida_{}".format(i), IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]))
            scales[j + 1 :] = scales[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, "ida_{}".format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class TableCenterNet(nn.Module):
    def __init__(self, backbone, heads, first_level, last_level, final_kernel, head_conv, out_channel=0):
        super(TableCenterNet, self).__init__()
        self.first_level = first_level
        self.last_level = last_level
        self.base = backbone
        channels = self.base.channels
        scales = [2**i for i in range(len(channels[self.first_level :]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level :], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level : self.last_level], [2**i for i in range(self.last_level - self.first_level)])

        self.heads = heads
        for head in self.heads:
            fc = self.get_head_layer(channels[self.first_level], head, head_conv, final_kernel)

            if "hm" in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self._fill_fc_weights(fc)

            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())

        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return [z]

    @staticmethod
    def _fill_fc_weights(layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def _parae_head(head):
        if isinstance(head, dict):
            return head["value"], head

        return head, {}

    def get_head_layer(self, in_channels, head, hidden_channels=-1, final_ksize=1):
        out_channels, config = self._parae_head(self.heads[head])

        if hidden_channels > 0:
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, out_channels, kernel_size=final_ksize, stride=1, padding=final_ksize // 2, bias=True),
            )
        else:
            return (nn.Conv2d(in_channels, out_channels, kernel_size=final_ksize, stride=1, padding=final_ksize // 2, bias=True),)
