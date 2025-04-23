#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
Version:
Autor: dreamy-xay
Date: 2024-10-25 17:41:00
LastEditors: dreamy-xay
LastEditTime: 2024-10-28 12:31:02
"""
import torch
from engine.table.decode import IndexQueryer, polygons_decode, is_group_ray, is_group, is_group_faster, find_near_corner_index, dist_square
from engine.table.utils import _tranpose_and_gather_feat
from shapely.geometry import Polygon


# def get_logic_coords(lc, span, center_indexes):
#     """
#     获取单元格的角点逻辑坐标

#     Returns:
#         - logic_coords: 单元格的角点逻辑坐标，(4)，其中 4 为 (start_col, end_col, start_row, end_row) 四个元素
#     """
#     one = torch.tensor(1.0, dtype=lc.dtype, device=lc.device)

#     # 获取单元格起始逻辑坐标
#     start_lcoords = _tranpose_and_gather_feat(lc, center_indexes)  # BxNx2
#     start_lcoords = torch.max(torch.round(start_lcoords), one)

#     # 获取跨度
#     cell_spans = _tranpose_and_gather_feat(span, center_indexes)  # BxNx2
#     cell_spans = torch.max(torch.floor(cell_spans), one)

#     end_lcoords = start_lcoords + cell_spans - 1

#     logic_coords = torch.cat((start_lcoords, end_lcoords), dim=2)

#     logic_coords[..., [1, 2]] = logic_coords[..., [2, 1]]

#     return logic_coords


def get_logic_coords(lc_logic, cell_span):
    """
    获取单元格的角点逻辑坐标

    Returns:
        - logic_coords: 单元格的角点逻辑坐标，(4)，其中 4 为 (start_col, end_col, start_row, end_row) 四个元素
    """
    # 获取单元格行列跨度
    col_span, row_span = max(1, int(torch.round(cell_span[0]))), max(1, int(torch.round(cell_span[1])))

    # 获取单元格的角点列行逻辑坐标
    col_logic_coords = []
    row_logic_coords = []
    for logic in lc_logic:
        col, row = max(1, int(torch.round(logic[0]))), max(1, int(torch.round(logic[1])))
        col_logic_coords.append(col)
        row_logic_coords.append(row)

    start_col, end_col, start_row, end_row = 0, 0, 0, 0

    # 计算单元格跨列信息
    if col_logic_coords[0] == col_logic_coords[3]:
        start_col = col_logic_coords[0]
        end_col = start_col + col_span - 1
    elif col_logic_coords[1] == col_logic_coords[2]:
        end_col = max(col_span + 1, col_logic_coords[1]) - 1
        start_col = end_col + 1 - col_span
    else:
        if torch.abs(lc_logic[0][0] - lc_logic[3][0]) <= torch.abs(lc_logic[1][0] - lc_logic[2][0]):
            start_col = max(1, int(torch.round((lc_logic[0][0] + lc_logic[3][0]) / 2.0)))
            end_col = start_col + col_span - 1
        else:
            end_col = max(col_span + 1, int(torch.round((lc_logic[1][0] + lc_logic[2][0]) / 2.0))) - 1
            start_col = end_col + 1 - col_span

    # 计算单元格跨行信息
    if row_logic_coords[0] == row_logic_coords[1]:
        start_row = row_logic_coords[0]
        end_row = start_row + row_span - 1
    elif row_logic_coords[2] == row_logic_coords[3]:
        end_row = max(row_span + 1, row_logic_coords[2]) - 1
        start_row = end_row + 1 - row_span
    else:
        if torch.abs(lc_logic[0][1] - lc_logic[1][1]) <= torch.abs(lc_logic[2][1] - lc_logic[3][1]):
            start_row = max(1, int(torch.round((lc_logic[0][1] + lc_logic[1][1]) / 2.0)))
            end_row = start_row + row_span - 1
        else:
            end_row = max(row_span + 1, int(torch.round((lc_logic[2][1] + lc_logic[3][1]) / 2.0))) - 1
            start_row = end_row + 1 - row_span

    return torch.Tensor([start_col, end_col, start_row, end_row])


def cells_decode(heatmap, reg, ct2cn, cn2ct, lc, span, center_k, corner_k, center_thresh, corner_thresh, corners=False):
    """
    单元格解码函数（仅支持 batch 为 1 时）

    Args:
        - heatmap: 热力图，(batch, 2, height, width)
        - reg: 中心点或角点的偏移向量图，(batch, 2, height, width)
        - ct2cn: 角点指向中心点的向量图，(batch, 8, height, width)
        - cn2ct: 中心点指向角点的向量图，(batch, 8, height, width)
        - lc: 角点逻辑坐标，(batch, 2, height, width)
        - span: 单元格跨度，(batch, 2, height, width)
        - center_k: 最大中心点数量
        - corner_k: 最大角点数量
        - center_thresh: 中心点阈值，中心点分数小于该阈值则不参与后续计算
        - corner_thresh: 角点阈值，角点分数小于该阈值则不参与后续计算
        - corners: 是否返回角点坐标

    Returns:
        - cells: 单元格，(batch, center_k, 8)，其中 center_k 为中心点数量，8 为单元格四个角点xy坐标，即左上、右上、左下、右下
        - cells_scores: 单元格分数，(batch, center_k, 1)，其中分数由高到低排序
        - cells_corner_count: 单元格角点优化次数，(batch, center_k, 2)，其中最后一个维度有两个次数，第一个为优化角点个数（最大为4），第二个为重复优化次数
        - logic_coords: 单元格逻辑坐标
    """

    # 获取中心点相关信息
    center_scores, center_indexes, center_xs, center_ys, center_polygons = polygons_decode(heatmap[:, 0:1, :, :], ct2cn, reg, K=center_k)

    # 获取角点相关信息
    corner_scores, corner_indexes, corner_xs, corner_ys, corner_polygons = polygons_decode(heatmap[:, 1:2, :, :], cn2ct, reg, K=corner_k)

    # 获取跨度
    cell_spans = _tranpose_and_gather_feat(span, center_indexes)

    # 获取逻辑坐标
    corner_logics = _tranpose_and_gather_feat(lc, corner_indexes)

    # 获取 cpu 状态下的 polygon
    if center_polygons.device.type != "cpu":
        center_polygons_cpu = center_polygons.cpu()
        corner_polygons_cpu = corner_polygons.cpu()
    else:
        center_polygons_cpu = center_polygons
        corner_polygons_cpu = corner_polygons

    # 索引查询器
    iq = IndexQueryer(center_polygons, corner_polygons, center_scores, corner_scores, center_thresh, corner_thresh)

    # 创建修正后的单元格
    corrected_cells = center_polygons.clone()

    # 创建单元格角点的修正个数和重复修正个数
    cells_corner_count = torch.zeros(center_polygons.shape[:-1] + (2,), dtype=torch.int32)

    # 创建单元格的逻辑坐标
    logic_coords = torch.zeros(center_polygons.shape[:-1] + (4,), dtype=torch.int32)

    # 遍历中心点
    for i in iq.center_indices:
        # 获取当前中心点对应的多边形（此处应是四边形）
        center_polygon = center_polygons[0, i, :].view(-1, 2)
        center_polygon_cpu = Polygon(center_polygons_cpu[0, i, :].view(-1, 2))

        # 获取当前中心点对应的单元格
        corrected_cell = corrected_cells[0, i, :].view(-1, 2)

        # 记录单元格角点修正个数和重复修正个数
        corner_count = 0
        repeat_corner_count = 0

        lc_logic = [None, None, None, None]

        # 遍历角点
        # for j in iq.corner_indices:
        for j in iq.query(i):
            # 获取当前角点对应的多边形（此处应是四边形）
            corner_polygon_cpu = corner_polygons_cpu[0, j, :].view(-1, 2)

            # 判断当前角点是否属于当前中心点对应的多边形
            if is_group_faster(center_polygon_cpu, corner_polygon_cpu):
                # 获取当前角点的坐标
                corner_x = corner_xs[0, j, 0]
                corner_y = corner_ys[0, j, 0]

                # 获取当前角点在多边形中的索引
                index = find_near_corner_index(center_polygon, corner_x, corner_y)

                # 获取被修正单元格指定角点
                corrected_cell_corner = corrected_cell[index]

                # 获取原始单元格的指定角点的坐标
                origin_corner_x = center_polygon[index][0]
                origin_corner_y = center_polygon[index][1]

                # 获取被修正单元格指定角点的坐标
                corrected_corner_x = corrected_cell_corner[0]
                corrected_corner_y = corrected_cell_corner[1]

                # 如果被修正单元格和原始单元格的指定角点相同，则直接修正，否则计算距离并修正距离最近的角点
                if corrected_corner_x == origin_corner_x and corrected_corner_y == origin_corner_y:
                    corner_count += 1
                    corrected_cell_corner[0] = corner_x
                    corrected_cell_corner[1] = corner_y
                    lc_logic[index] = corner_logics[0, j]
                elif dist_square(origin_corner_x, origin_corner_y, corrected_corner_x, corrected_corner_y) >= dist_square(origin_corner_x, origin_corner_y, corner_x, corner_y):
                    repeat_corner_count += 1
                    corrected_cell_corner[0] = corner_x
                    corrected_cell_corner[1] = corner_y
                    lc_logic[index] = corner_logics[0, j]

        cells_corner_count[0, i, 0] = corner_count
        cells_corner_count[0, i, 1] = repeat_corner_count

        # 更新当前单元格的逻辑坐标
        for j in range(4):
            if lc_logic[j] is None:
                x, y = corrected_cell[j].tolist()
                x = 0 if x < 0 else (lc.shape[3] - 1 if x >= lc.shape[3] else int(x))
                y = 0 if y < 0 else (lc.shape[2] - 1 if y >= lc.shape[2] else int(y))
                lc_logic[j] = lc[0, :, y, x]

        logic_coords[0, i, :] = get_logic_coords(lc_logic, cell_spans[0, i].clone())

    if corners:
        return corrected_cells, center_scores, cells_corner_count, logic_coords.to(reg.device), torch.cat([corner_xs, corner_ys, corner_scores], dim=2)

    return corrected_cells, center_scores, cells_corner_count, logic_coords.to(reg.device)


def simple_cells_decode(heatmap, reg, ct2cn, cn2ct, lc, span, center_k, corner_k, center_thresh, corners=False):
    """
    简单单元格解码函数（仅支持 batch 为 1 时）

    Args:
        - heatmap: 热力图，(batch, 2, height, width)
        - reg: 中心点或角点的偏移向量图，(batch, 2, height, width)
        - ct2cn: 角点指向中心点的向量图，(batch, 8, height, width)
        - cn2ct: 中心点指向角点的向量图，(batch, 8, height, width)
        - lc: 角点逻辑坐标，(batch, 2, height, width)
        - span: 单元格跨度，(batch, 2, height, width)
        - center_k: 最大中心点数量
        - corner_k: 最大角点数量
        - center_thresh: 中心点阈值，中心点分数小于该阈值则不参与后续计算
        - corners: 是否返回角点坐标

    Returns:
        - cells: 单元格，(batch, center_k, 8)，其中 center_k 为中心点数量，8 为单元格四个角点xy坐标，即左上、右上、左下、右下
        - cells_scores: 单元格分数，(batch, center_k, 1)，其中分数由高到低排序
        - logic_coords: 单元格逻辑坐标
    """

    # 获取中心点相关信息
    center_scores, center_indexes, center_xs, center_ys, center_polygons = polygons_decode(heatmap[:, 0:1, :, :], ct2cn, reg, K=center_k)

    # 获取跨度
    cell_spans = _tranpose_and_gather_feat(span, center_indexes)

    # 获取 center_polygon 和 corner_polygon 的数目
    num_center_polygons = center_polygons.shape[1]

    # 创建单元格的逻辑坐标
    logic_coords = torch.zeros(center_polygons.shape[:-1] + (4,), dtype=torch.int32)

    # 遍历中心点
    for i in range(num_center_polygons):
        # 如果中心点分数低于阈值，则退出，因为 center_scores 是重高到低排序的
        if center_scores[0, i, 0] < center_thresh:
            break

        # 获取当前中心点对应的单元格
        cell = center_polygons[0, i, :].view(-1, 2)

        lc_logic = [None, None, None, None]

        # 更新当前单元格的逻辑坐标
        for j in range(4):
            if lc_logic[j] is None:
                x, y = cell[j].tolist()
                x = 0 if x < 0 else (lc.shape[3] - 1 if x >= lc.shape[3] else int(x))
                y = 0 if y < 0 else (lc.shape[2] - 1 if y >= lc.shape[2] else int(y))
                lc_logic[j] = lc[0, :, y, x]

        logic_coords[0, i, :] = get_logic_coords(lc_logic, cell_spans[0, i])

    if corners:
        # 获取角点相关信息
        corner_scores, corner_indexes, corner_xs, corner_ys, corner_polygons = polygons_decode(heatmap[:, 1:2, :, :], cn2ct, reg, K=corner_k)
        return center_polygons, center_scores, logic_coords.to(reg.device), torch.cat([corner_xs, corner_ys, corner_scores], dim=2)

    return center_polygons, center_scores, logic_coords.to(reg.device)
