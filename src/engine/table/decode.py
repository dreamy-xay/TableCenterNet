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
from shapely.geometry import Polygon, Point
from shapely.affinity import scale
from .utils import _nms, _tranpose_and_gather_feat, _topk
from utils.utils import BoxesFinder

class IndexQueryer:
    def __init__(self, center_polygons, corner_polygons, center_scores, corner_scores, center_thresh, corner_thresh):
        # 获取 center_polygon 和 corner_polygon 的数目
        num_center_polygons = center_polygons.shape[1]
        num_corner_polygons = corner_polygons.shape[1]
        
        # 角点多边形构造区间缓存，并提供查询
        self.center_indices = valid_center_indices = (center_scores.view(num_center_polygons) >= center_thresh).nonzero().squeeze(dim=-1)
        if valid_center_indices.numel() > 0:
            self.corner_indices = valid_corner_indices = (corner_scores.view(num_corner_polygons) >= corner_thresh).nonzero().squeeze(dim=-1)
            self.exist_corners = update_cell_corners = valid_corner_indices.numel() > 0
            if update_cell_corners:
                vcenter_polygons = center_polygons[0, valid_center_indices, :]
                vcorner_polygons = corner_polygons[0, valid_corner_indices, :]
                self.boxes_finder = BoxesFinder(
                    torch.stack((vcorner_polygons[:, 0::2].amin(dim=-1), vcorner_polygons[:, 0::2].amax(dim=-1)), dim=1).tolist(),
                    torch.stack((vcorner_polygons[:, 1::2].amin(dim=-1), vcorner_polygons[:, 1::2].amax(dim=-1)), dim=1).tolist(),
                    torch.stack((vcenter_polygons[:, 0::2].amin(dim=-1), vcenter_polygons[:, 0::2].amax(dim=-1)), dim=1).tolist(),
                    torch.stack((vcenter_polygons[:, 1::2].amin(dim=-1), vcenter_polygons[:, 1::2].amax(dim=-1)), dim=1).tolist(),
                )
    
    def query(self, index):
        return sorted(list(self.boxes_finder.query(index))) if self.exist_corners else []

def is_group_ray(center_polygon, corner_polygon):
    """
    判断多个点是否在多边形内

    参数：
        - center_polygon: 形状为 (4, 2) 的 NumPy 数组，表示多边形顶点坐标
        - corner_polygon: 形状为 (4, 2) 的 NumPy 数组，表示需要判断的点坐标

    返回：
        - is_group: corner_polygon 中是否存在一个点在 center_polygon 中
    """
    # num_points = center_polygon.shape[0]  # 多边形顶点数 (4)

    # for point in corner_polygon:
    #     crossings = 0
    #     x, y = point

    #     j = num_points - 1 # j为前一个顶点
    #     for i in range(num_points): 
    #         ix = center_polygon[i, 0]
    #         iy = center_polygon[i, 1]
    #         jx = center_polygon[j, 0]
    #         jy = center_polygon[j, 1]
            
    #         # 判断点在两个x之间 且以点垂直y轴向上做射线
    #         if ((ix > x) != (jx > x)) and (x > (jx - ix) * (y - iy) / (jy - iy) + ix):
    #             crossings += 1
            
    #         # 更新j
    #         j = i
        
    #     # 奇数次交点表示在内部，偶数次交点表示在外部
    #     if crossings & 1:
    #         return True

    # return False
    # 将多边形顶点与判断点扩展为矩阵
    center_x = center_polygon[:, 0]
    center_y = center_polygon[:, 1]

    # 使用torch.roll将多边形顶点顺序偏移一个位置，模拟顶点的“前一个”位置
    # 例如， center_polygon[i] 和 center_polygon[j] 用于计算交点
    prev_x = torch.roll(center_x, 1)
    prev_y = torch.roll(center_y, 1)

    # 计算每条边的dx, dy
    dx = center_x - prev_x
    dy = center_y - prev_y + 1e-6

    # 计算每个点是否与每条边相交
    # 将 corner_polygon 扩展成 (num_points, 1) 的形状，便于批量运算
    x = corner_polygon[:, 0]
    y = corner_polygon[:, 1]

    # 判断每个点是否与多边形的每条边相交
    t1 = ((center_x > x[:, None]) != (prev_x > x[:, None]))  # 判断x坐标是否跨越边界
    t2 = (x[:, None] > (dx * (y[:, None] - center_y) / dy + center_x))  # 判断射线与边的交点

    # 计算交点次数
    crossings = torch.sum(t1 & t2, axis=1)

    # 如果交点次数是奇数，表示该点在多边形内部
    return torch.any(crossings % 2 != 0)

def is_group_faster(center_polygon, corner_polygon):
    """判断角点多边形的某个角点是否在中心点多边形内部，要求 center_polygon 和 corner_polygon 输入为 (N, 2) 的张量"""
    # 遍历角点多边形的全部个角点是否在中心点多边形内部
    for i in range(corner_polygon.size(0)):
        pt = Point(corner_polygon[i])
        if pt.within(center_polygon):
            return True

    return False

def is_group(center_polygon, corner_polygon, scale_factor=1.0):
    """判断角点多边形的某个角点是否在中心点多边形内部，要求 center_polygon 和 corner_polygon 输入为 (N, 2) 的张量"""
    # 获取中心点多边形的边界框的坐标最值
    ctp_xmin, ctp_xmax, ctp_ymin, ctp_ymax = center_polygon[:, 0].min(), center_polygon[:, 0].max(), center_polygon[:, 1].min(), center_polygon[:, 1].max()
    # 获取角点多边形的边界框的坐标最值
    cnp_xmin, cnp_xmax, cnp_ymin, cnp_ymax = corner_polygon[:, 0].min(), corner_polygon[:, 0].max(), corner_polygon[:, 1].min(), corner_polygon[:, 1].max()

    # 如果角点多边形不存在某个角点在多边形内部，则返回False
    if ctp_xmin > cnp_xmax or cnp_xmin > ctp_xmax or ctp_ymin > cnp_ymax or cnp_ymin > ctp_ymax:
        return False

    # 创建中心点多边形
    _center_polygon = Polygon(center_polygon)
    if scale_factor < 1.0:
        # 计算多边形的中心点
        centroid = _center_polygon.centroid
        # 以中心点为基础缩小多边形，缩小比例为 scale
        _center_polygon = scale(_center_polygon, xfact=scale_factor, yfact=scale_factor, origin=centroid)

    # 遍历角点多边形的全部个角点是否在中心点多边形内部
    for i in range(corner_polygon.size(0)):
        pt = Point(corner_polygon[i])
        if pt.within(_center_polygon):
            return True

    return False


def dist_square(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def find_near_corner_index(polygon, pt_x, pt_y):
    """查询临近角点索引，要求 polygon 输入为 (N, 2) 的张量"""

    # 获取当前中心点对应的多边形坐标
    xs = polygon[:, 0]
    ys = polygon[:, 1]

    # 计算平方距离
    distances_square = (xs - pt_x) ** 2 + (ys - pt_y) ** 2

    # 返回最近角点的索引
    return torch.argmin(distances_square)


def polygons_decode(heatmap, vec, reg, K=400):
    batch = heatmap.size(0)

    # 获取热力图的峰值区域（四周低中间高）
    heatmap = _nms(heatmap)[0]

    # 获取热力图分数排名的前K个点（中心点或角点）
    scores, indexes, _, ys, xs = _topk(heatmap, K=K)

    # 分数格式转换
    scores = scores.view(batch, K, 1)

    # 获取中心点或角点坐标的偏移量
    point_offset = _tranpose_and_gather_feat(reg, indexes)

    # 获取中心点或角点的坐标
    xs = xs.view(batch, K, 1) + point_offset[:, :, 0:1]
    ys = ys.view(batch, K, 1) + point_offset[:, :, 1:2]

    # 获取指向中心点或角点的回归框的向量
    polygons_vec = _tranpose_and_gather_feat(vec, indexes)

    # 获取中心点或角点的回归框（xy为中心点时：四个点是单元格左上、右上、左下、右下的角点坐标；xy为角点时：四个点是左上、右上、左下、右下单元格的回归中心点）的坐标
    polygons = torch.cat(
        [
            xs - polygons_vec[..., 0:1],
            ys - polygons_vec[..., 1:2],
            xs - polygons_vec[..., 2:3],
            ys - polygons_vec[..., 3:4],
            xs - polygons_vec[..., 4:5],
            ys - polygons_vec[..., 5:6],
            xs - polygons_vec[..., 6:7],
            ys - polygons_vec[..., 7:8],
        ],
        dim=2,
    )

    return scores, indexes, xs, ys, polygons


def cells_decode(heatmap, reg, ct2cn, cn2ct, center_k, corner_k, center_thresh, corner_thresh, corners=False):
    """
    单元格解码函数（仅支持 batch 为 1 时）

    Args:
        - heatmap: 热力图，(batch, 2, height, width)
        - reg: 中心点或角点的偏移向量图，(batch, 2, height, width)
        - ct2cn: 角点指向中心点的向量图，(batch, 8, height, width)
        - cn2ct: 中心点指向角点的向量图，(batch, 8, height, width)
        - center_k: 最大中心点数量
        - corner_k: 最大角点数量
        - center_thresh: 中心点阈值，中心点分数小于该阈值则不参与后续计算
        - corner_thresh: 角点阈值，角点分数小于该阈值则不参与后续计算
        - corners: 是否返回角点坐标

    Returns:
        - cells: 单元格，(batch, center_k, 8)，其中 center_k 为中心点数量，8 为单元格四个角点xy坐标，即左上、右上、左下、右下
        - cells_scores: 单元格分数，(batch, center_k, 1)，其中分数由高到低排序
        - cells_corner_count: 单元格角点优化次数，(batch, center_k, 2)，其中最后一个维度有两个次数，第一个为优化角点个数（最大为4），第二个为重复优化次数
    """

    # 获取中心点相关信息
    center_scores, center_indexes, center_xs, center_ys, center_polygons = polygons_decode(heatmap[:, 0:1, :, :], ct2cn, reg, K=center_k)

    # 获取角点相关信息
    corner_scores, corner_indexes, corner_xs, corner_ys, corner_polygons = polygons_decode(heatmap[:, 1:2, :, :], cn2ct, reg, K=corner_k)

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

    # 遍历中心点
    for i in iq.center_indices:
        # 获取当前中心点对应的多边形（此处应是四边形）
        center_polygon = center_polygons[0, i, :].view(-1, 2)

        center_polygon_cpu = center_polygons_cpu[0, i, :].view(-1, 2)

        # 获取当前中心点对应的单元格
        corrected_cell = corrected_cells[0, i, :].view(-1, 2)

        # 记录单元格角点修正个数和重复修正个数
        corner_count = 0
        repeat_corner_count = 0

        # 遍历角点
        # for j in iq.corner_indices:
        for j in iq.query(i):
            # 获取当前角点对应的多边形（此处应是四边形）
            corner_polygon_cpu = corner_polygons_cpu[0, j, :].view(-1, 2)

            # 判断当前角点是否属于当前中心点对应的多边形
            if is_group(center_polygon_cpu, corner_polygon_cpu):
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
                elif dist_square(origin_corner_x, origin_corner_y, corrected_corner_x, corrected_corner_y) >= dist_square(origin_corner_x, origin_corner_y, corner_x, corner_y):
                    repeat_corner_count += 1
                    corrected_cell_corner[0] = corner_x
                    corrected_cell_corner[1] = corner_y

        cells_corner_count[0, i, 0] = corner_count
        cells_corner_count[0, i, 1] = repeat_corner_count

    if corners:
        return corrected_cells, center_scores, cells_corner_count, torch.cat([corner_xs, corner_ys, corner_scores], dim=2)

    return corrected_cells, center_scores, cells_corner_count


def logic_coords_decode(lc, cells):
    col_lc = lc[:, 0]
    row_lc = lc[:, 1]

    results = []

    for i, batch in enumerate(cells):
        batch_col_lc = col_lc[i]
        batch_row_lc = row_lc[i]
        batch_result = []
        height, width = batch_col_lc.shape
        for cell in batch:
            x1, y1, x2, y2, x3, y3, x4, y4 = tuple(map(int, cell.cpu().tolist()))

            if not (0 < x1 < width and 0 < x2 < width and 0 < x3 < width and 0 < x4 < width and 0 < y1 < height and 0 < y2 < height and 0 < y3 < height and 0 < y4 < height):
                batch_result.append([0, 0, 0, 0])
                continue

            start_col = torch.floor((torch.round(batch_col_lc[y1, x1]) + torch.round(batch_col_lc[y4, x4])) / 2.0)
            end_col = torch.floor((torch.round(batch_col_lc[y2, x2]) + torch.round(batch_col_lc[y3, x3])) / 2.0) - 1
            start_row = torch.floor((torch.round(batch_row_lc[y1, x1]) + torch.round(batch_row_lc[y2, x2])) / 2.0)
            end_row = torch.floor((torch.round(batch_row_lc[y3, x3]) + torch.round(batch_row_lc[y4, x4])) / 2.0) - 1
            # start_col = torch.round((batch_col_lc[y1, x1] + batch_col_lc[y4, x4]) / 2.0)
            # end_col = torch.round((batch_col_lc[y2, x2] + batch_col_lc[y3, x3]) / 2.0) - 1, start_col
            # start_row = torch.round((batch_row_lc[y1, x1] + batch_row_lc[y2, x2]) / 2.0)
            # end_row = torch.round((batch_row_lc[y3, x3] + batch_row_lc[y4, x4]) / 2.0) - 1, start_row

            end_col = torch.max(start_col, end_col)
            end_row = torch.max(start_row, end_row)

            batch_result.append([start_col, end_col, start_row, end_row])

        results.append(batch_result)

    return torch.Tensor(results).to(cells.device)
