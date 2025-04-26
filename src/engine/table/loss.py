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
from torch import nn
import torch.nn.functional as F
from .utils import _tranpose_and_gather_feat, _sigmoid


class FocalLoss(nn.Module):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, out, target):
        return self._neg_loss(out, target)

    @staticmethod
    def _neg_loss(pred, gt):
        """Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        else:
            return -(pos_loss + neg_loss) / num_pos


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, ind, mask, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum") / (mask.sum() + 1e-4)
        return loss


class VecPairLoss(nn.Module):
    EPS = 1e-4

    def __init__(self):
        super(VecPairLoss, self).__init__()

    def forward(self, ct2cn, ct_ind, ct_mask, ct2cn_gt, cn2ct, cn_ind, cn_mask, cn2ct_gt, ct_cn_ind):
        # Obtain the error relationship between the midpoint and the corner point
        ct2cn_pred = _tranpose_and_gather_feat(ct2cn, ct_ind)  # BM8: B->batch, M->number of center point
        cn2ct_pred = _tranpose_and_gather_feat(cn2ct, cn_ind)  # BN8: B->batch, N->number of corner point

        # Temporary caches cn2ct_pred and cn2ct_gt are used to calculate the third part of the loss
        cn2ct_pred_temp = cn2ct_pred
        cn2ct_gt_temp = cn2ct_gt

        # batch, number of center point, number of corner point
        B, M, N = ct2cn_pred.size(0), ct2cn_pred.size(1), cn2ct_pred.size(1)

        # Convert the pred and gt of CN2CT to the shape of CT2CN
        ct_cn_ind = ct_cn_ind.unsqueeze(2).expand(B, 4 * M, 2)
        cn2ct_pred = cn2ct_pred.view(B, 4 * N, 2).gather(1, ct_cn_ind).view(B, M, 8)
        cn2ct_gt = cn2ct_gt.view(B, 4 * N, 2).gather(1, ct_cn_ind).view(B, M, 8)

        # Convert the mask of ct to the shape of ct2cn_pred and get the number of center points
        ct_mask = ct_mask.unsqueeze(2).expand_as(ct2cn_pred).float()
        num_ct = ct_mask.sum() + self.EPS  # Prevent dividing by zero

        # Convert the mask of cn to the shape of cn2ct_pred_temp
        cn_mask = cn_mask.unsqueeze(2).expand_as(cn2ct_pred_temp)

        # Calculate delta
        delta = (torch.abs(ct2cn_pred - ct2cn_gt) + torch.abs(cn2ct_pred - cn2ct_gt)) / (torch.abs(ct2cn_gt) + self.EPS)
        # delta = torch.min(delta * delta, torch.tensor(1.0))
        delta = torch.min(delta, torch.tensor(1.0))

        # Calculate weight
        weight = torch.sin(1.570796 * delta)  # Trigonoconvex function (torch.cos(1.570796 * (delta - 1.0)))

        # Calculate the vector pair loss for ct2cn and cn2ct
        ct2cn_loss = F.l1_loss(ct2cn_pred * ct_mask * weight, ct2cn_gt * ct_mask * weight, reduction="sum") / num_ct
        cn2ct_loss = F.l1_loss(cn2ct_pred * ct_mask * weight, cn2ct_gt * ct_mask * weight, reduction="sum") / num_ct

        # Calculate the loss of vector pairs where there shouldn't be
        invalid_vec_mask = cn2ct_gt_temp == 0  # If there is no data from the corner to the center point, an invalid vector mask is obtained
        invalid_vec_cn_mask = (invalid_vec_mask == cn_mask).float()  # Get a mask of an invalid vector with corners
        invalid_vec_cn_loss = F.l1_loss(cn2ct_pred_temp * invalid_vec_cn_mask, cn2ct_gt_temp * invalid_vec_cn_mask, reduction="sum") / (invalid_vec_cn_mask.sum() + self.EPS)

        return ct2cn_loss, 0.5 * cn2ct_loss, 0.2 * invalid_vec_cn_loss


class LogicCoordLoss(torch.nn.Module):
    EPS = 1e-4

    def __init__(self, crood_loss_weights):
        super(LogicCoordLoss, self).__init__()

        self._loss_weights = crood_loss_weights

    """
    def forward(self, coord, coord_gt, coord_mask, lc_ind, lc_span, ct_mask):
        # * Calculate the loss of logical coordinates
        coord_loss = F.l1_loss(coord * coord_mask, coord_gt * coord_mask, reduction="sum") / (coord_mask.sum() + self.EPS)

        return coord_loss, torch.tensor(0.0, device=coord_loss.device)
    """

    def forward(self, coord, coord_gt, coord_mask, lc_ind, lc_span, ct_mask):
        B = lc_span.size(0)
        N = lc_span.size(1)  # Number of cells

        # * Initialize the data
        # Get the logical coordinates and mask of the corresponding corner point of each cell
        coords_pred = _tranpose_and_gather_feat(coord, lc_ind.view(B, N * 4)).view(B, N, 4, 2)  # Bx4Nx2 -> BxNx4x2
        cols_pred = coords_pred[..., 0]  # BxNx4
        rows_pred = coords_pred[..., 1]  # BxNx4

        # Get the span value of each cell as well as the mask
        span_mask = ct_mask.unsqueeze(2).expand(B, N, 2).float()  # BxNx2
        num_span_mask = span_mask.sum() + self.EPS

        # * Calculate the loss of logical coordinates
        coord_loss = F.l1_loss(coord * coord_mask, coord_gt * coord_mask, reduction="sum") / (coord_mask.sum() + self.EPS)

        # * Calculate the weight required for span loss
        col_span_diff_pred = cols_pred[..., [1, 2]] - cols_pred[..., [0, 3]]  # BxNx2
        row_span_diff_pred = rows_pred[..., [3, 2]] - rows_pred[..., [0, 1]]  # BxNx2
        col_span_gt = lc_span[..., 0].unsqueeze(2).expand(B, N, 2)  # BxN -> BxNx2
        row_span_gt = lc_span[..., 1].unsqueeze(2).expand(B, N, 2)  # BxN -> BxNx2

        # * Calculate the error loss of the logical coordinate span
        # Calculate the error loss of the span of the logical coordinate column
        col_span_diff_loss = F.l1_loss(col_span_diff_pred * span_mask, col_span_gt * span_mask, reduction="sum") / num_span_mask
        # Calculate the error loss of the logical coordinate row span
        row_span_diff_loss = F.l1_loss(row_span_diff_pred * span_mask, row_span_gt * span_mask, reduction="sum") / num_span_mask
        # Statistical loss of the total logical coordinate span error
        span_diff_loss = col_span_diff_loss + row_span_diff_loss

        return coord_loss, span_diff_loss


class TableLoss(torch.nn.Module):
    loss_stats = ["loss", "hm", "reg", "ct2cn", "cn2ct", "icn2ct", "lc", "lsd"]

    def __init__(self, loss_weights):
        super(TableLoss, self).__init__()
        self.hm_weight, self.reg_weight, self.ct2cn_weight, self.cn2ct_weight, self.lc_weight = loss_weights

        self.hm_crit = FocalLoss()
        self.reg_crit = RegL1Loss()
        self.vec_pair_crit = VecPairLoss()
        self.lc_crit = LogicCoordLoss((1, 1, 1, 1))

    def forward(self, outputs, batch):
        output = outputs[-1]
        output["hm"] = _sigmoid(output["hm"])

        hm_loss = self.hm_crit(output["hm"], batch["hm"])

        reg_loss = self.reg_crit(output["reg"], batch["reg_ind"], batch["reg_mask"], batch["reg"])

        ct2cn_loss, cn2ct_loss, invalid_cn2ct_loss = self.vec_pair_crit(
            output["ct2cn"], batch["ct_ind"], batch["ct_mask"], batch["ct2cn"], output["cn2ct"], batch["cn_ind"], batch["cn_mask"], batch["cn2ct"], batch["ct_cn_ind"]
        )

        lc_coord_loss, lc_span_diff_loss = self.lc_crit(output["lc"], batch["lc"], batch["lc_mask"], batch["lc_ind"], batch["lc_span"], batch["ct_mask"])
        lc_loss = lc_coord_loss + lc_span_diff_loss

        loss = self.hm_weight * hm_loss + self.reg_weight * reg_loss + self.ct2cn_weight * ct2cn_loss + self.cn2ct_weight * (cn2ct_loss + invalid_cn2ct_loss) + self.lc_weight * lc_loss

        loss_stats = {"loss": loss, "hm": hm_loss, "reg": reg_loss, "ct2cn": ct2cn_loss, "cn2ct": cn2ct_loss, "icn2ct": invalid_cn2ct_loss, "lc": lc_coord_loss, "lsd": lc_span_diff_loss}

        return loss, loss_stats
