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

    def forward(self, lc, span, lc_gt, span_gt, ct_ind, ct_mask):
        # 获取每个单元格的起始逻辑坐标损失
        lc_pred = _tranpose_and_gather_feat(lc, ct_ind)  # BxNx2
        lc_mask = ct_mask.unsqueeze(2).expand_as(lc_pred).float()  # BxNx2
        num_lc_mask = lc_mask.sum() + self.EPS
        lc_loss = F.l1_loss(lc_pred * lc_mask, lc_gt * lc_mask, reduction="sum") / num_lc_mask
        
        # 获取每个单元格的跨度损失
        span_pred = _tranpose_and_gather_feat(span, ct_ind)  # BxNx2
        span_mask = lc_mask  # BxNx2
        num_span_mask = num_lc_mask
        span_loss = F.l1_loss(span_pred * span_mask, span_gt * span_mask, reduction="sum") / num_span_mask

        return lc_loss, span_loss


class STableLoss(torch.nn.Module):
    loss_stats = ["loss", "hm", "reg", "ct2cn", "cn2ct", "icn2ct", "lc", "sp"]

    def __init__(self, loss_weights):
        super(STableLoss, self).__init__()
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

        lc_coord_loss, lc_span_loss = self.lc_crit(output["lc"], output["sp"], batch["lc_coords"], batch["lc_span"], batch["ct_ind"], batch["ct_mask"])

        lc_loss = lc_coord_loss + lc_span_loss

        loss = self.hm_weight * hm_loss + self.reg_weight * reg_loss + self.ct2cn_weight * ct2cn_loss + self.cn2ct_weight * (cn2ct_loss + invalid_cn2ct_loss) + self.lc_weight * lc_loss

        loss_stats = {"loss": loss, "hm": hm_loss, "reg": reg_loss, "ct2cn": ct2cn_loss, "cn2ct": cn2ct_loss, "icn2ct": invalid_cn2ct_loss, "lc": lc_coord_loss, "sp": lc_span_loss}

        return loss, loss_stats
