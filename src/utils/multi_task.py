#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-11-25 10:31:52
LastEditors: dreamy-xay
LastEditTime: 2025-01-16 18:19:08
'''
import torch
import torch.nn.functional as F

class DynamicWeightAveraging():
    """
    动态权重平均 (DWA)。

    参数:
        T (float, 默认=2.0): Softmax 温度。
    """
    def __init__(self, T=2.0):
        self.T = T  # 温度参数
        self.loss_buffer = [None, None]

    def __call__(self, losses):
        """
        计算给定损失的动态权重。

        参数:
            losses (torch.Tensor): 形状为 (task_num,) 的张量，包含每个任务的损失。

        返回:
            torch.Tensor: 形状为 (task_num,) 的张量，包含每个任务的计算权重。
        """
        # 使用新的损失值更新损失缓冲区
        self.loss_buffer[0] = self.loss_buffer[1]
        self.loss_buffer[1] = losses

        # 如果没有之前的损失数据，则所有任务的权重相同
        if self.loss_buffer[0] is None:
            weights = torch.ones_like(losses)
        else:
            # 基于最后两次的损失计算每个任务的权重
            w_i = (self.loss_buffer[1] / (self.loss_buffer[0] + 1e-8))  # 加一个小值避免除以零
            weights = len(losses) * F.softmax(w_i / self.T, dim=-1)  # 应用 softmax 并缩放

        return weights