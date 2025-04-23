#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-11-04 21:05:22
LastEditors: dreamy-xay
LastEditTime: 2024-11-04 21:05:36
"""
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Polygon
from shapely.ops import triangulate
from shapely.validation import make_valid
from shapely.affinity import scale
import math
import cv2


def split_polygon(polygon):
    """将多边形分割为多个三角形，支持凹多边形"""
    try:
        # 创建一个凹多边形
        cur_polygon = Polygon([(x, y) for x, y, _ in polygon])
    except:
        return []

    # 如果多边形无效，则尝试进行修复
    if not cur_polygon.is_valid:
        cur_polygon = make_valid(cur_polygon)

        if not cur_polygon.is_valid:
            return []
    try:
        # 使用Shapely的triangulate进行三角剖分
        triangles = triangulate(cur_polygon)
    except:
        return []

    point_map = {}
    for x, y, v in polygon:
        point_map[(x, y)] = v

    # 将三角形转化为带有v值的多边形
    new_triangles = []
    for triangle in triangles:
        try:
            if not cur_polygon.contains(triangle):
                continue
        except:
            continue
        # 获取三角形的所有顶点坐标
        triangle_points = list(triangle.exterior.coords)
        # 为每个顶点添加v值
        try:
            triangle_v = [(x, y, point_map[(x, y)]) for (x, y) in triangle_points[:-1]]
        except:
            continue
        new_triangles.append(triangle_v)

    return new_triangles


def polygon_area(points):
    """计算多边形的面积"""
    x, y = zip(*points)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def interpolate_polygons(polygons, img_size, processes=["sort"]):
    """
    插值多边形

    参数：
        polygons: 多边形列表，每个多边形是一个包含(x, y, v)的列表，v表示要插的值
        img_size: (height, width)
        process: 预处理类型，可选值有"sort"和"split"，"split"支持凹多边形分解为多个三角形，"sort"按面积排序
    返回：
        final_image: 插值后的图像
        mask: 掩码，用于区分插值区域和背景区域
    """
    # 初始化一个全为0的图像和全为False的掩码
    final_image = np.zeros(img_size, dtype=np.float32)
    mask = np.zeros(img_size, dtype=bool)

    processed_polygons = polygons
    # 预处理多边形
    for process in processes:
        if process == "sort":
            # 计算每个多边形的面积
            areas = [polygon_area([(x, y) for x, y, _ in polygon]) for polygon in processed_polygons]

            # 按面积排序
            processed_polygons = [polygon for _, polygon in sorted(zip(areas, processed_polygons))]
        elif process == "split":
            # 将多边形分割为多个三角形
            new_processed_polygons = []
            for polygon in processed_polygons:
                new_processed_polygons.extend(split_polygon(polygon))
            processed_polygons = new_processed_polygons

    # 对每个多边形进行插值
    for polygon in processed_polygons:
        # 获取 bbox
        x_min = math.floor(min(x for x, y, _ in polygon))
        y_min = math.floor(min(y for x, y, _ in polygon))
        x_max = math.ceil(max(x for x, y, _ in polygon))
        y_max = math.ceil(max(y for x, y, _ in polygon))
        bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

        # 提取点和值
        points = np.array([(x - bbox[0], y - bbox[1]) for x, y, _ in polygon], dtype=np.float32)
        values = np.array([v for _, _, v in polygon], dtype=np.float32)

        # 创建网格
        x, y = np.meshgrid(np.arange(bbox[2]), np.arange(bbox[3]))
        grid_points = np.vstack((x.ravel(), y.ravel())).T

        # 插值
        try:
            interpolated_values = griddata(points, values, grid_points, method="linear", fill_value=-1)
        except:
            continue

        # 将插值结果叠加到最终图像上，并更新掩码
        for (i, j), value in zip(grid_points, interpolated_values):
            i += bbox[0]
            j += bbox[1]
            if value >= 0 and not mask[j, i]:
                final_image[j, i] = value
                mask[j, i] = True

    return final_image, mask


class Line:
    def __init__(self, points):
        if len(points) < 2:
            self.a, self.b, self.c = 0, 0, 0
        else:
            points = np.asarray(points, dtype=np.float32)
            a, b, c = self.fit_line(points[:, :2])
            self.a = a
            self.b = b
            self.c = c

    def instersect(self, other):
        if self.a == other.a and self.b == other.b:
            raise ValueError("Lines are parallel and do not intersect.")
        divisor = self.a * other.b - other.a * self.b
        x = (self.b * other.c - other.b * self.c) / divisor
        y = (other.a * self.c - self.a * other.c) / divisor
        return x, y

    @property
    def equation(self):
        return self.a, self.b, self.c

    @staticmethod
    def fit_line(points, eps=1e-6):
        # 使用 cv2.fitLine 拟合直线
        line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

        # 解析输出
        vx, vy, x0, y0 = line.flatten()

        # 计算直线方程 ax + by + c = 0 的参数
        a = float(-vy if abs(vy) > eps else 0)
        b = float(vx if abs(vx) > eps else 0)
        c = float(-(a * x0 + b * y0))
        return a, b, c


def split_unicom_area(cells, adjacent_thresh=1):
    # 并查集分割
    num_cells = len(cells)
    parent = [i for i in range(num_cells)]

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    def get_polygon_scale_kargs(polygon, value):
        # 获取多边形的边界框 (Bounding Box)
        minx, miny, maxx, maxy = polygon.bounds

        # 计算宽度和高度
        width = maxx - minx
        height = maxy - miny

        # 计算缩放因子
        xfact = (width + 2 * value) / width
        yfact = (height + 2 * value) / height

        return {"geom": polygon, "xfact": xfact, "yfact": yfact, "origin": polygon.centroid}

    polygons = [Polygon([point[:2] for point in cell]) for cell in cells]
    scale_polygons = [scale(**get_polygon_scale_kargs(polygon, adjacent_thresh)) for polygon in polygons]

    for i in range(num_cells):
        for j in range(i + 1, num_cells):
            if scale_polygons[i].intersects(polygons[j]):
                union(i, j)

    # 将并查集结果转换为连通区域
    connected_components = {}
    for i in range(num_cells):
        root = find(i)
        if root not in connected_components:
            connected_components[root] = []
        connected_components[root].append(i)

    return list(connected_components.values())

def simple_interpolate_cells(cells, img_size):
    """
    在图像中插值单元格

    参数：
        cells: 单元格的顶点列表，每个顶点是一个元组 (x, y, col, row)
        img_size: 图像的尺寸，用于初始化图像和掩码
    返回值：
        final_image: 插值后的图像
        mask: 掩码，用于标记插值后的像素
    """
    cols = []
    rows = []
    
    # 获取单元格框
    for cell in cells:
        cols.append([point[:3] for point in cell])
        rows.append([[point[0], point[1], point[3]] for point in cell])

    # 生成插值图
    col_final_image, col_mask = interpolate_polygons(cols, img_size, ["sort"])
    row_final_image, row_mask = interpolate_polygons(rows, img_size, ["sort"])

    return np.stack([col_final_image, row_final_image], axis=0), np.stack([col_mask, row_mask], axis=0)

def interpolate_cells(cells, img_size, adjacent_thresh=1):
    """
    在图像中插值单元格

    参数：
        cells: 单元格的顶点列表，每个顶点是一个元组 (x, y, col, row)
        img_size: 图像的尺寸，用于初始化图像和掩码
        adjacent_thresh: 邻近阈值，用于确定单元格的边界
    返回值：
        final_image: 插值后的图像
        mask: 掩码，用于标记插值后的像素
    """
    cols = []
    rows = []
    # 获取新的行列框
    for area_cells_index in split_unicom_area(cells, adjacent_thresh):
        area_cells = [cells[i] for i in area_cells_index]
        # 计算最大列和行
        max_cols = max(max(point[2] for point in cell) for cell in area_cells)
        max_rows = max(max(point[3] for point in cell) for cell in area_cells)

        col_points_list = [[] for _ in range(max_cols)]
        row_points_list = [[] for _ in range(max_rows)]

        for cell in area_cells:
            for point in cell:
                col_points_list[point[2] - 1].append(point)
                row_points_list[point[3] - 1].append(point)

        # 排序
        for points in col_points_list:
            points.sort(key=lambda pt: pt[1])
        for points in row_points_list:
            points.sort(key=lambda pt: pt[0])

        # 获取边界线
        upper_line, upper_bound = Line(row_points_list[0]), row_points_list[0]
        lower_line, lower_bound = Line(row_points_list[-1]), row_points_list[-1]
        left_line, left_bound = Line(col_points_list[0]), col_points_list[0]
        right_line, right_bound = Line(col_points_list[-1]), col_points_list[-1]

        # 补全列坐标
        for col, points in enumerate(col_points_list):
            if len(points) < 2:
                continue

            cur_line = Line(points)
            col += 1
            if points[0][3] != 1:
                try:
                    x, y = upper_line.instersect(cur_line)
                    if 0 < x < img_size[1] and 0 < y < img_size[0]:
                        instersection = [x, y, col, 1]
                        points.append(instersection)
                        upper_bound.append(instersection)
                except:
                    pass
            if points[-1][3] != max_rows:
                try:
                    x, y = lower_line.instersect(cur_line)
                    if 0 < x < img_size[1] and 0 < y < img_size[0]:
                        instersection = [x, y, col, max_rows]
                        points.append(instersection)
                        lower_bound.append(instersection)
                except:
                    pass

        # 补全行坐标
        for row, points in enumerate(row_points_list):
            if len(points) < 2:
                continue

            cur_line = Line(points)
            row += 1
            if points[0][2] != 1:
                try:
                    x, y = left_line.instersect(cur_line)
                    if 0 < x < img_size[1] and 0 < y < img_size[0]:
                        instersection = [x, y, 1, row]
                        points.append(instersection)
                        left_bound.append(instersection)
                except:
                    pass
            if points[-1][2] != max_cols:
                try:
                    x, y = right_line.instersect(cur_line)
                    if 0 < x < img_size[1] and 0 < y < img_size[0]:
                        instersection = [x, y, max_cols, row]
                        points.append(instersection)
                        right_bound.append(instersection)
                except:
                    pass

        # 重新排序
        for i in range(max_cols):
            col_points_list[i] = [point[:3] for point in col_points_list[i]]
            col_points_list[i].sort(key=lambda pt: pt[1])
        for i in range(max_rows):
            row_points_list[i] = [[*point[:2], point[3]] for point in row_points_list[i]]
            row_points_list[i].sort(key=lambda pt: pt[0])

        for i in range(1, max_cols):
            cols.append(col_points_list[i - 1])
            cols[-1].extend(col_points_list[i][::-1])

        for i in range(1, max_rows):
            rows.append(row_points_list[i - 1])
            rows[-1].extend(row_points_list[i][::-1])

    # 生成插值图
    col_final_image, col_mask = interpolate_polygons(cols, img_size, ["split"])
    row_final_image, row_mask = interpolate_polygons(rows, img_size, ["split"])

    return np.stack([col_final_image, row_final_image], axis=0), np.stack([col_mask, row_mask], axis=0)


def multitable_interpolate_cells(cells, img_size):
    """
    在图像中插值单元格

    参数：
        cells: 单元格的顶点列表 [corner1, ..., table_id]，每个顶点是一个元组 (x, y, col, row)
        img_size: 图像的尺寸，用于初始化图像和掩码
    返回值：
        final_image: 插值后的图像
        mask: 掩码，用于标记插值后的像素
    """
    cols = []
    rows = []

    # 分割表格区域
    tables = {}
    for cell in cells:
        table_id = cell[-1]
        if table_id not in tables:
            tables[table_id] = []

        tables[table_id].append(cell[:-1])

    # 获取新的行列框
    for area_cells in tables.values():
        # 计算最大列和行
        max_cols = max(max(point[2] for point in cell) for cell in area_cells)
        max_rows = max(max(point[3] for point in cell) for cell in area_cells)

        col_points_list = [[] for _ in range(max_cols)]
        row_points_list = [[] for _ in range(max_rows)]

        for cell in area_cells:
            for point in cell:
                col_points_list[point[2] - 1].append(point)
                row_points_list[point[3] - 1].append(point)

        # 排序
        for points in col_points_list:
            points.sort(key=lambda pt: pt[1])
        for points in row_points_list:
            points.sort(key=lambda pt: pt[0])

        # 获取边界线
        upper_line, upper_bound = Line(row_points_list[0]), row_points_list[0]
        lower_line, lower_bound = Line(row_points_list[-1]), row_points_list[-1]
        left_line, left_bound = Line(col_points_list[0]), col_points_list[0]
        right_line, right_bound = Line(col_points_list[-1]), col_points_list[-1]

        # 补全列坐标
        for col, points in enumerate(col_points_list):
            if len(points) < 2:
                continue

            cur_line = Line(points)
            col += 1
            if points[0][3] != 1:
                try:
                    x, y = upper_line.instersect(cur_line)
                    if 0 < x < img_size[1] and 0 < y < img_size[0]:
                        instersection = [x, y, col, 1]
                        points.append(instersection)
                        upper_bound.append(instersection)
                except:
                    pass
            if points[-1][3] != max_rows:
                try:
                    x, y = lower_line.instersect(cur_line)
                    if 0 < x < img_size[1] and 0 < y < img_size[0]:
                        instersection = [x, y, col, max_rows]
                        points.append(instersection)
                        lower_bound.append(instersection)
                except:
                    pass

        # 补全行坐标
        for row, points in enumerate(row_points_list):
            if len(points) < 2:
                continue

            cur_line = Line(points)
            row += 1
            if points[0][2] != 1:
                try:
                    x, y = left_line.instersect(cur_line)
                    if 0 < x < img_size[1] and 0 < y < img_size[0]:
                        instersection = [x, y, 1, row]
                        points.append(instersection)
                        left_bound.append(instersection)
                except:
                    pass
            if points[-1][2] != max_cols:
                try:
                    x, y = right_line.instersect(cur_line)
                    if 0 < x < img_size[1] and 0 < y < img_size[0]:
                        instersection = [x, y, max_cols, row]
                        points.append(instersection)
                        right_bound.append(instersection)
                except:
                    pass

        # 重新排序
        for i in range(max_cols):
            col_points_list[i] = [point[:3] for point in col_points_list[i]]
            col_points_list[i].sort(key=lambda pt: pt[1])
        for i in range(max_rows):
            row_points_list[i] = [[*point[:2], point[3]] for point in row_points_list[i]]
            row_points_list[i].sort(key=lambda pt: pt[0])

        for i in range(1, max_cols):
            cols.append(col_points_list[i - 1])
            cols[-1].extend(col_points_list[i][::-1])

        for i in range(1, max_rows):
            rows.append(row_points_list[i - 1])
            rows[-1].extend(row_points_list[i][::-1])

    # 生成插值图
    col_final_image, col_mask = interpolate_polygons(cols, img_size, ["split"])
    row_final_image, row_mask = interpolate_polygons(rows, img_size, ["split"])

    return np.stack([col_final_image, row_final_image], axis=0), np.stack([col_mask, row_mask], axis=0)


def line_interpolation(img, p1, p2, value):
    """在图像中插值两个点之间的像素"""
    # 获取两个点的坐标
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]

    # 计算步长
    num_step = max(abs(x2 - x1), abs(y2 - y1))
    step_x = (x2 - x1) / (num_step + 1e-6)
    step_y = (y2 - y1) / (num_step + 1e-6)

    # 获取图像宽高
    img_height, img_width = img.shape[:2]

    # 根据距离插值出所有中间点
    for i in range(num_step + 1):
        x = round(x1 + i * step_x)
        y = round(y1 + i * step_y)

        # 保证坐标在图像范围内
        if 0 <= x < img_width and 0 <= y < img_height:
            img[y, x] = value  # 插值

    return img


def interpolate_cells_edges(cells, img_size, extend_pixel=8):
    """
    在图像中插值单元格的边缘

    参数：
        cells: 单元格的顶点列表，每个顶点是一个元组 (x, y, col, row)
        img_size: 图像的尺寸，用于初始化图像和掩码
        extend_pixel: 扩展像素数，用于扩展单元格的边缘
    返回值：
        final_image: 插值后的图像
        mask: 掩码，用于标记插值后的像素
    """
    # 初始化一个全为0的图像和全为False的掩码
    col_final_image = np.zeros(img_size, dtype=np.float32)
    row_final_image = np.zeros(img_size, dtype=np.float32)
    mask = np.zeros(img_size, dtype=np.uint8)

    for area_cells_index in split_unicom_area(cells):
        area_cells = [cells[i] for i in area_cells_index]
        # 计算最大列和行
        max_cols = max(max(point[2] for point in cell) for cell in area_cells)
        max_rows = max(max(point[3] for point in cell) for cell in area_cells)

        col_points_list = [[] for _ in range(max_cols)]
        row_points_list = [[] for _ in range(max_rows)]

        for cell in area_cells:
            for point in cell:
                col_points_list[point[2] - 1].append(point)
                row_points_list[point[3] - 1].append(point)

        # 排序
        for points in col_points_list:
            points.sort(key=lambda pt: pt[1])
        for points in row_points_list:
            points.sort(key=lambda pt: pt[0])

        # 获取边界线
        upper_line, upper_bound = Line(row_points_list[0]), row_points_list[0]
        lower_line, lower_bound = Line(row_points_list[-1]), row_points_list[-1]
        left_line, left_bound = Line(col_points_list[0]), col_points_list[0]
        right_line, right_bound = Line(col_points_list[-1]), col_points_list[-1]

        # 补全列坐标
        for col, points in enumerate(col_points_list):
            cur_line = Line(points)
            col += 1
            if points[0][3] != 1:
                instersection = [*upper_line.instersect(cur_line), col, 1]
                points.append(instersection)
                upper_bound.append(instersection)
            if points[-1][3] != max_rows:
                instersection = [*lower_line.instersect(cur_line), col, max_rows]
                points.append(instersection)
                lower_bound.append(instersection)

        # 补全行坐标
        for row, points in enumerate(row_points_list):
            cur_line = Line(points)
            row += 1
            if points[0][2] != 1:
                instersection = [*left_line.instersect(cur_line), 1, row]
                points.append(instersection)
                left_bound.append(instersection)
            if points[-1][2] != max_cols:
                instersection = [*right_line.instersect(cur_line), max_cols, row]
                points.append(instersection)
                right_bound.append(instersection)

        # 重新排序
        for points in col_points_list:
            for point in points:
                point[0], point[1] = round(point[0]), round(point[1])
            points.sort(key=lambda pt: pt[1])
        for points in row_points_list:
            for point in points:
                point[0], point[1] = round(point[0]), round(point[1])
            points.sort(key=lambda pt: pt[0])

        area = [point[:2] for point in [*upper_bound, *right_bound, *lower_bound[::-1], *left_bound[::-1]]]
        area = np.array([area], dtype=np.int32)

        # 更新mask
        cv2.fillPoly(mask, area, 1.0)
        cv2.polylines(mask, area, True, 1.0, thickness=extend_pixel * 2 + 1)

        # 列插值
        for col, points in enumerate(col_points_list):
            col += 1
            for j in range(len(points) - 1):
                line_interpolation(col_final_image, points[j], points[j + 1], col)

        # 行插值
        for row, points in enumerate(row_points_list):
            row += 1
            for j in range(len(points) - 1):
                line_interpolation(row_final_image, points[j], points[j + 1], row)

    return np.stack([col_final_image, row_final_image], axis=0), np.stack([mask, mask], axis=0)
