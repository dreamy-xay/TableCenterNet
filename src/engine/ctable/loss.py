#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 15:43:36
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 15:49:12
"""
import torch
import torch.nn.functional as F
from engine.table.loss import FocalLoss, RegL1Loss, VecPairLoss
from engine.table.utils import _tranpose_and_gather_feat, _sigmoid


class LogicCoordLoss(torch.nn.Module):
    EPS = 1e-4

    def __init__(self):
        super(LogicCoordLoss, self).__init__()

    @staticmethod
    def _calc_weights(output1, output2, target):
        # 计算 delta
        delta = (torch.abs(output1 - target) + torch.abs(output2 - target)) * 5.0  # diff * 10 / 2
        delta = torch.min(delta, torch.tensor(1.0))

        # 计算 weight
        weight = torch.sin(1.570796 * delta)

        return weight

    """
    def forward(self, coord, span, lc_coords, lc_span, ct_ind, ct_mask):
        B = lc_span.size(0)
        N = lc_span.size(1)  # 单元格数目

        # * 初始化数据
        # 获取每个单元格对应角点的逻辑坐标以及索引
        coord_ind, cols_gt, rows_gt = lc_coords[..., 0], lc_coords[..., 1], lc_coords[..., 2]  # BxNx4, BxNx4, BxNx4

        # 获取每个单元格对应角点的逻辑坐标以及掩模
        coords_pred = _tranpose_and_gather_feat(coord, coord_ind.view(B, N * 4)).view(B, N, 4, 2)  # Bx4Nx2 -> BxNx4x2
        cols_pred = coords_pred[..., 0]  # BxNx4
        rows_pred = coords_pred[..., 1]  # BxNx4
        coord_mask = ct_mask.unsqueeze(2).expand(B, N, 4).float()  # BxNx4
        num_coord_mask = coord_mask.sum() + self.EPS

        # 获取每个单元格的跨度值以及掩模
        span_pred = _tranpose_and_gather_feat(span, ct_ind)  # BxNx2
        span_mask = ct_mask.unsqueeze(2).expand(B, N, 2).float()  # BxNx2
        num_span_mask = span_mask.sum() + self.EPS

        # * 计算逻辑坐标损失
        # 计算逻辑坐标列误差损失
        col_coord_loss = F.l1_loss(cols_pred * coord_mask, cols_gt * coord_mask, reduction="sum") / num_coord_mask
        # 计算逻辑坐标行误差损失
        row_coord_loss = F.l1_loss(rows_pred * coord_mask, rows_gt * coord_mask, reduction="sum") / num_coord_mask
        # 统计总的逻辑坐标误差损失
        coord_loss = col_coord_loss + row_coord_loss

        # * 计算跨度损失所需权重
        col_span_diff_pred = cols_pred[..., [1, 2]] - cols_pred[..., [0, 3]]  # BxNx2
        row_span_diff_pred = rows_pred[..., [3, 2]] - rows_pred[..., [0, 1]]  # BxNx2
        # col_span_pred = span_pred[..., 0].unsqueeze(2).expand(B, N, 2)  # BxN -> BxNx2
        # row_span_pred = span_pred[..., 1].unsqueeze(2).expand(B, N, 2)  # BxN -> BxNx2
        col_span_gt = lc_span[..., 0].unsqueeze(2).expand(B, N, 2)  # BxN -> BxNx2
        row_span_gt = lc_span[..., 1].unsqueeze(2).expand(B, N, 2)  # BxN -> BxNx2
        # col_span_weight = self._calc_weights(col_span_pred, col_span_diff_pred, col_span_gt)  # BxNx2
        # row_span_weight = self._calc_weights(row_span_pred, row_span_diff_pred, row_span_gt)  # BxNx2
        # span_weight = torch.stack([(col_span_weight[..., 0] + col_span_weight[..., 1]) / 2.0, (row_span_weight[..., 0] + row_span_weight[..., 1]) / 2.0], dim=-1)  # BxNx2

        # * 计算逻辑坐标跨度误差损失
        # 计算逻辑坐标列跨度误差损失
        # col_span_diff_loss = F.l1_loss(col_span_diff_pred * span_mask * col_span_weight, col_span_gt * span_mask * col_span_weight, reduction="sum") / num_span_mask
        col_span_diff_loss = F.l1_loss(col_span_diff_pred * span_mask, col_span_gt * span_mask, reduction="sum") / num_span_mask
        # 计算逻辑坐标行跨度误差损失
        # row_span_diff_loss = F.l1_loss(row_span_diff_pred * span_mask * row_span_weight, row_span_gt * span_mask * row_span_weight, reduction="sum") / num_span_mask
        row_span_diff_loss = F.l1_loss(row_span_diff_pred * span_mask, row_span_gt * span_mask, reduction="sum") / num_span_mask
        # 统计总的逻辑坐标跨度误差损失
        span_diff_loss = col_span_diff_loss + row_span_diff_loss

        # * 计算跨度损失
        # span_loss = F.l1_loss(span_pred * span_mask * span_weight, lc_span * span_mask * span_weight, reduction="sum") / num_span_mask
        span_loss = F.l1_loss(span_pred * span_mask, lc_span * span_mask, reduction="sum") / num_span_mask

        return coord_loss, span_diff_loss, span_loss
    """
    
    def forward(self, coord, span, lc_coords, lc_span, ct_ind, ct_mask):
        B = lc_span.size(0)
        N = lc_span.size(1)  # 单元格数目

        # * 初始化数据
        # 获取每个单元格对应角点的逻辑坐标以及索引
        coord_ind, cols_gt, rows_gt = lc_coords[..., 0], lc_coords[..., 1], lc_coords[..., 2]  # BxNx4, BxNx4, BxNx4

        # 获取每个单元格对应角点的逻辑坐标以及掩模
        coords_pred = _tranpose_and_gather_feat(coord, coord_ind.view(B, N * 4)).view(B, N, 4, 2)  # Bx4Nx2 -> BxNx4x2
        cols_pred = coords_pred[..., 0]  # BxNx4
        rows_pred = coords_pred[..., 1]  # BxNx4
        coord_mask = ct_mask.unsqueeze(2).expand(B, N, 4).float()  # BxNx4
        num_coord_mask = coord_mask.sum() + self.EPS

        # 获取每个单元格的跨度值以及掩模
        span_pred = _tranpose_and_gather_feat(span, ct_ind)  # BxNx2
        span_mask = ct_mask.unsqueeze(2).expand(B, N, 2).float()  # BxNx2
        num_span_mask = span_mask.sum() + self.EPS

        # * 计算逻辑坐标损失
        # 计算逻辑坐标列误差损失
        col_coord_loss = F.l1_loss(cols_pred * coord_mask, cols_gt * coord_mask, reduction="sum") / num_coord_mask
        # 计算逻辑坐标行误差损失
        row_coord_loss = F.l1_loss(rows_pred * coord_mask, rows_gt * coord_mask, reduction="sum") / num_coord_mask
        # 统计总的逻辑坐标误差损失
        coord_loss = col_coord_loss + row_coord_loss

        # * 计算跨度损失
        span_loss = F.l1_loss(span_pred * span_mask, lc_span * span_mask, reduction="sum") / num_span_mask

        return coord_loss, torch.tensor(0.0, device=coord_loss.device), span_loss


class CTableLoss(torch.nn.Module):
    loss_stats = ["loss", "hm", "reg", "ct2cn", "cn2ct", "icn2ct", "lc", "lsd", "ls"]

    def __init__(self, loss_weights):
        super(CTableLoss, self).__init__()
        self.hm_weight, self.reg_weight, self.ct2cn_weight, self.cn2ct_weight, self.lc_weight = loss_weights

        self.hm_crit = FocalLoss()
        self.reg_crit = RegL1Loss()
        self.vec_pair_crit = VecPairLoss()
        self.lc_crit = LogicCoordLoss()

    def forward(self, outputs, batch):
        output = outputs[-1]
        output["hm"] = _sigmoid(output["hm"])

        hm_loss = self.hm_crit(output["hm"], batch["hm"])

        reg_loss = self.reg_crit(output["reg"], batch["reg_ind"], batch["reg_mask"], batch["reg"])

        ct2cn_loss, cn2ct_loss, invalid_cn2ct_loss = self.vec_pair_crit(
            output["ct2cn"], batch["ct_ind"], batch["ct_mask"], batch["ct2cn"], output["cn2ct"], batch["cn_ind"], batch["cn_mask"], batch["cn2ct"], batch["ct_cn_ind"]
        )

        lc_coord_loss, lc_span_diff_loss, lc_span_loss = self.lc_crit(output["lc"], output["sp"], batch["lc_coords"], batch["lc_span"], batch["ct_ind"], batch["ct_mask"])

        lc_loss = lc_coord_loss + lc_span_diff_loss + lc_span_loss

        loss = self.hm_weight * hm_loss + self.reg_weight * reg_loss + self.ct2cn_weight * ct2cn_loss + self.cn2ct_weight * (cn2ct_loss + invalid_cn2ct_loss) + self.lc_weight * lc_loss

        loss_stats = {"loss": loss, "hm": hm_loss, "reg": reg_loss, "ct2cn": ct2cn_loss, "cn2ct": cn2ct_loss, "icn2ct": invalid_cn2ct_loss, "lc": lc_coord_loss, "lsd": lc_span_diff_loss, "ls": lc_span_loss}

        return loss, loss_stats
