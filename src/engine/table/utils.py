#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 15:45:49
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 15:52:46
"""
import torch
import numpy as np
from shapely.geometry import Polygon


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)  # 防止数值溢出
    return y


def _gather_feat(feat, ind):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)

    keep = (hmax == heat).float()
    return heat * keep, keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    device = scores.device

    topk_inds = topk_inds % (torch.Tensor([height]).to(device, torch.int64) * torch.Tensor([width]).to(device, torch.int64))
    topk_ys = (topk_inds / torch.Tensor([width]).to(device)).int().float()
    topk_xs = (topk_inds % torch.Tensor([width]).to(device, torch.int64)).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _pnms(detections, iou_thresh, overlap=False):
    # 多边形数目
    num_polygons = detections.shape[0]

    # 多边形数组
    polygons = []
    for i in range(num_polygons):
        polygons.append(Polygon([detections[i][0:2], detections[i][2:4], detections[i][4:6], detections[i][6:8]]))

    # 多边形面积数组
    areas = np.zeros(num_polygons)
    # 多边形相交面积数组
    intersect_areas = np.zeros((num_polygons, num_polygons))

    # 计算多边形面积和相交面积
    for i in range(0, num_polygons):
        polygon_i = polygons[i]
        areas[i] = polygon_i.area

        for j in range(i + 1, num_polygons):
            polygon_j = polygons[j]
            intersect_polygon = polygon_i.intersection(polygon_j)
            intersect_areas[i][j] = intersect_areas[j][i] = intersect_polygon.area

    # 无效多边形标签数组
    inviad_targets = [False] * len(polygons)

    # 计算多边形之间的交并比
    for i in range(0, num_polygons):
        polygon_i_area = areas[i]
        polygon_i_score = detections[i][8]
        for j in range(i + 1, num_polygons):
            polygon_j_area = areas[j]
            polygon_j_score = detections[j][8]
            iou = intersect_areas[i][j] / (min(polygon_i_area, polygon_j_area) if overlap else (polygon_i_area + polygon_j_area - intersect_areas[i][j]))
            if iou > iou_thresh:
                if polygon_i_score > polygon_j_score:
                    # if polygon_i_area > polygon_j_area:
                    inviad_targets[j] = True
                else:
                    inviad_targets[i] = True

    # 删除无效多边形
    results = []
    for i in range(num_polygons):
        if not inviad_targets[i]:
            results.append(detections[i])

    return np.array(results)
