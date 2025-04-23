#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
Version:
Autor: dreamy-xay
Date: 2024-10-22 09:53:56
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 10:39:45
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}

# add file proxy for github
# github_file_url_proxy = "https://ghproxy.net/"  # github_file_url_proxy = "https://ghproxy.net/"
# for net, v in model_urls.items():
#     model_urls[net] = github_file_url_proxy + v


BN_MOMENTUM = 0.1


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module("conv", torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module("bn", torch.nn.BatchNorm2d(out_planes, momentum=BN_MOMENTUM))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.0):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, in_channel=32, **kwargs):
        super().__init__()
        self.in_channel = in_channel
        self.channels = [self.in_channel]
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2**i_layer
            self.channels.append(embed_dim)
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        y = [x]
        for stage in self.stages:
            x = stage(x)
            y.append(x)
        return y

    def load_pretrained_model(self, name="starnet_s1"):
        if name.endswith(".tar"):
            checkpoint = torch.load(name)
        else:
            url = model_urls[name]
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")

        num_classes = len(checkpoint["state_dict"][list(checkpoint["state_dict"].keys())[-1]])
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.head = nn.Linear(self.in_channel, num_classes)

        self.load_state_dict(checkpoint["state_dict"])

        delattr(self, "norm")
        delattr(self, "head")


def starnet_s1(pretrained=True, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        model.load_pretrained_model("starnet_s1")
    return model


def starnet_s2(pretrained=True, **kwargs):
    model = StarNet(32, [1, 2, 6, 2], **kwargs)
    if pretrained:
        model.load_pretrained_model("starnet_s2")
    return model


def starnet_s3(pretrained=True, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], **kwargs)
    if pretrained:
        model.load_pretrained_model("starnet_s3")
    return model


def starnet_s4(pretrained=True, **kwargs):
    model = StarNet(32, [3, 3, 12, 5], **kwargs)
    if pretrained:
        model.load_pretrained_model("starnet_s4")
    return model


# very small networks #
def starnet_s050(**kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)


def starnet_s100(**kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)


def starnet_s150(**kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)


# plus
def starnet_s3_plus(**kwargs):
    return StarNet(32, [2, 2, 8, 4], in_channel=64, **kwargs)
