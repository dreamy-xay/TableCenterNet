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
from .TableCenterNet import TableCenterNet
from nn.backbone.StarNet import *


def get_startsr(heads, head_conv=256, arch="s3"):
    backbone = globals()[f"starnet_{arch}"](pretrained=True)
    model = TableCenterNet(backbone, heads, 1, 4, final_kernel=1, head_conv=head_conv)
    return model
