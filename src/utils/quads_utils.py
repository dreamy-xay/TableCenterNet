#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 获取单元格结构
Version: 
Autor: dreamy-xay
date: 2024-07-09
LastEditors: dreamy-xay
LastEditTime: 2024-07-09
"""
from typing import List, Tuple, Optional, Set, Any
from typing_extensions import Literal, Self
from dataclasses import dataclass
import math


# * 区间对象
@dataclass
class Interval:
    start: int  # 区间开始
    end: int  # 区间结束


# * 跨行跨列信息
@dataclass
class CrossInfo:
    row: int  # 跨行信息
    col: int  # 跨列信息


# * 关系图
@dataclass
class RelationMap:
    top: List["QuadInfo"]
    right: List["QuadInfo"]
    bottom: List["QuadInfo"]
    left: List["QuadInfo"]


# * 四边形（单元格）信息
@dataclass
class QuadInfo:
    # 数据
    quad: List["Point"]

    # 索引信息
    index: int

    # 结构信息
    row_range: Interval
    col_range: Interval
    cross: CrossInfo

    # 图信息
    forward: RelationMap  # 前向（包含或等价）关系图
    backward: RelationMap  # 反向（被包含）关系图

    # 缓存信息
    cache: List["QuadInfo"]


# * 点对象
class Point:
    def __init__(self, point: Tuple[int, int]) -> None:
        self.x = point[0]
        self.y = point[1]

    def is_adjacent(self, other: "Point", threshold: int = 10) -> bool:
        """
        判断两个点是否相邻

        Autor: dreamy-xay
        Date: 2024-07-09
        """
        return abs(self.x - other.x) < threshold and abs(self.y - other.y) < threshold

    def distance(self, line_point1: "Point", line_point2: "Point") -> float:
        """
        计算点到直线的距离

        Autor: dreamy-xay
        Date: 2024-07-09
        """
        # 计算直线一般方程
        # Ax + By + C = 0
        A = line_point2.y - line_point1.y
        B = line_point1.x - line_point2.x
        C = line_point2.x * line_point1.y - line_point1.x * line_point2.y

        # 计算点到直线距离
        # d = |Ax + By + C| / sqrt(A^2 + B^2)
        return abs(A * self.x + B * self.y + C) / math.sqrt(A**2 + B**2)

    def middle(self, other: "Point") -> "Point":
        """
        计算两个点的中点

        Autor: dreamy-xay
        Date: 2024-07-09
        """
        return Point(((self.x + other.x) // 2, (self.y + other.y) // 2))


# 并查集
class DisjointSet:
    def __init__(self, array: List[Any]) -> None:
        self.array = array
        self.size = len(array)
        self.parent = [i for i in range(self.size)]

    def find(self, x: int) -> int:
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> Self:  #
        """
        将两个元素合并到一个集合中

        Autor: dreamy-xay
        Date: 2024-07-09
        """
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root != y_root:
            self.parent[y_root] = x_root

        return self

    def get_all_collections(self) -> List[List[Any]]:
        """
        （合并完成后）获取所有集合

        Autor: dreamy-xay
        Date: 2024-07-09
        """
        all_collections: List[List[Any]] = []
        collection_index = {}
        for i in range(self.size):
            if self.find(i) == i:
                collection_index[i] = len(all_collections)
                all_collections.append([self.array[i]])
        for i in range(self.size):
            if self.parent[i] != i:
                all_collections[collection_index[self.parent[i]]].append(self.array[i])

        return all_collections


def find_continuous_quad(quads_info: List[QuadInfo], start_quad_info: QuadInfo, end_quad_info: QuadInfo, next_cache: List[int]) -> List[QuadInfo]:
    """
    查找一个起始四边形到另一个终止四边形中间所有的连续四边形列表（含起始四边形和终止四边形）

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    # 初始化数据
    temp_quads_info: List[QuadInfo] = [start_quad_info]
    temp_quads_keys = set()

    # 暴力迭代查询起始四边形到终止四边形中间所有的连续四边形列表
    while True:
        temp_quads_keys.add(temp_quads_info[-1].index)
        temp_quad_info: Optional[QuadInfo] = None

        # 查找到下一个四边形
        next_index = next_cache[temp_quads_info[-1].index]
        if next_index != -1:
            temp_quad_info = quads_info[next_index]

        if temp_quad_info is None or temp_quad_info.index == end_quad_info.index:
            break

        # 无限死循环了，直接退出
        if temp_quad_info.index in temp_quads_keys:
            temp_quads_info = [start_quad_info]
            break

        # 加入列表
        temp_quads_info.append(temp_quad_info)

    # 加入终止四边形
    temp_quads_info.append(end_quad_info)

    # 查询结束，返回结果
    return temp_quads_info


def build_sub_relation_map(quads_info: List[QuadInfo], dir: Literal["left", "right", "top", "bottom"], adjacent_threshold: float, strict_mode: bool = True) -> None:
    """
    构建单方向的前向和后向关系图

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    # 初始化数据
    start_point_index_pair: Tuple[int, int]
    end_point_index_pair: Tuple[int, int]
    reverse_dir: Literal["left", "right", "top", "bottom"]
    if dir == "left":
        start_point_index_pair = (0, 1)
        end_point_index_pair = (3, 2)
        reverse_dir = "right"
    elif dir == "right":
        start_point_index_pair = (1, 0)
        end_point_index_pair = (2, 3)
        reverse_dir = "left"
    elif dir == "top":
        start_point_index_pair = (0, 3)
        end_point_index_pair = (1, 2)
        reverse_dir = "bottom"
    else:
        start_point_index_pair = (3, 0)
        end_point_index_pair = (2, 1)
        reverse_dir = "top"

    # 四边形列表长度
    num_quads = len(quads_info)

    # 取出初始化数据
    start_index1, start_index2 = start_point_index_pair
    end_index1, end_index2 = end_point_index_pair

    # 建立 next 缓存
    next_cache = None

    def build_next_cache(next_cache):  # next 缓存只缓存一次，仅有需要时缓存
        if next_cache is not None:
            return next_cache
        next_cache = [-1] * num_quads
        for i in range(num_quads):
            for j in range(num_quads):
                if i != j and quads_info[i].quad[end_index2].is_adjacent(quads_info[j].quad[start_index2], adjacent_threshold):
                    next_cache[quads_info[i].index] = j
                    break
        return next_cache

    ### 单向关系图的构建
    for i in range(num_quads):
        quad_info = quads_info[i]

        # 查找开始矩形框
        start_quad_info: Optional[QuadInfo] = None
        for j in range(num_quads):
            if i != j and quad_info.quad[start_index1].is_adjacent(quads_info[j].quad[start_index2], adjacent_threshold):
                start_quad_info = quads_info[j]
                break

        # 查找结束矩形框
        end_quad_info: Optional[QuadInfo] = None
        for j in range(num_quads):
            if i != j and quad_info.quad[end_index1].is_adjacent(quads_info[j].quad[end_index2], adjacent_threshold):
                end_quad_info = quads_info[j]

        # 构建关系图
        if (start_quad_info is not None) and (end_quad_info is not None):
            dir_relation_map: List[QuadInfo] = getattr(quad_info.forward, dir)
            if start_quad_info.index == end_quad_info.index:
                dir_relation_map.append(start_quad_info)
            else:
                next_cache = build_next_cache(next_cache)  # 只会建立一次
                temp_quads_info = find_continuous_quad(quads_info, start_quad_info, end_quad_info, next_cache)
                dir_relation_map.extend(temp_quads_info)
                for temp_quad_info in temp_quads_info:
                    reverse_dir_relation_map: List[QuadInfo] = getattr(temp_quad_info.backward, reverse_dir)
                    reverse_dir_relation_map.append(quad_info)
        elif not strict_mode:
            if start_quad_info is not None:
                dir_relation_map: List[QuadInfo] = getattr(quad_info.forward, dir)
                dir_relation_map.append(start_quad_info)
            elif end_quad_info is not None:
                dir_relation_map: List[QuadInfo] = getattr(quad_info.forward, dir)
                dir_relation_map.append(end_quad_info)


def build_relation_map(quads: List[List[Tuple[int, int]]], adjacent_threshold: float = 10, strict_mode: bool = True) -> List[QuadInfo]:
    """
    构建四个方向的前向和后向关系图

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    num_quads = len(quads)

    # 初始化四边形列表信息
    quads_info = [
        QuadInfo(
            quad=[Point(point) for point in quads[i]],
            index=i,
            row_range=Interval(-1, -1),
            col_range=Interval(-1, -1),
            cross=CrossInfo(-1, -1),
            forward=RelationMap(top=[], right=[], bottom=[], left=[]),
            backward=RelationMap(top=[], right=[], bottom=[], left=[]),
            cache=[],
        )
        for i in range(num_quads)
    ]

    # 按照左上角y坐标排序
    quads_info.sort(key=lambda quad_info: quad_info.quad[0].y)

    # 左关系图的构建
    build_sub_relation_map(quads_info, "left", adjacent_threshold, strict_mode)
    # 右关系图的构建
    build_sub_relation_map(quads_info, "right", adjacent_threshold, strict_mode)

    # 按照左上角x坐标排序
    quads_info.sort(key=lambda quad_info: quad_info.quad[0].x)

    # 上关系图的构建
    build_sub_relation_map(quads_info, "top", adjacent_threshold, strict_mode)
    # 下关系图的构建
    build_sub_relation_map(quads_info, "bottom", adjacent_threshold, strict_mode)

    return quads_info


def deep_search_row_cross(quad_info: QuadInfo, search_start_stack: List[QuadInfo]) -> int:
    """
    正向深度搜索四边形跨行信息

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    if len(quad_info.forward.right) > 0:
        total_cross_row = 0

        for next_quad_info in quad_info.forward.right:
            total_cross_row += deep_search_row_cross(next_quad_info, search_start_stack)

        quad_info.cross.row = max(total_cross_row, quad_info.cross.row)
    else:
        if len(quad_info.backward.right) > 0:
            if len(search_start_stack) == 0 or search_start_stack[-1].index != quad_info.backward.right[0].index:
                search_start_stack.append(quad_info.backward.right[0])

        quad_info.cross.row = max(1, quad_info.cross.row)

    return quad_info.cross.row


def deep_search_col_cross(quad_info: QuadInfo, search_start_stack: List[QuadInfo]) -> int:
    """
    正向深度搜索四边形跨列信息

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    if len(quad_info.forward.bottom) > 0:
        total_cross_col = 0

        for next_quad_info in quad_info.forward.bottom:
            total_cross_col += deep_search_col_cross(next_quad_info, search_start_stack)

        quad_info.cross.col = max(total_cross_col, quad_info.cross.col)
    else:
        if len(quad_info.backward.bottom) > 0:
            if len(search_start_stack) == 0 or search_start_stack[-1].index != quad_info.backward.bottom[0].index:
                search_start_stack.append(quad_info.backward.bottom[0])

        quad_info.cross.col = max(1, quad_info.cross.col)

    return quad_info.cross.col


def reverse_deep_search_row_cross(quad_info: QuadInfo, search_start_stack: List[QuadInfo]) -> int:
    """
    反向深度搜索四边形跨行信息

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    if len(quad_info.forward.left) > 0:
        total_cross_row = 0

        for next_quad_info in quad_info.forward.left:
            total_cross_row += reverse_deep_search_row_cross(next_quad_info, search_start_stack)

        quad_info.cross.row = max(total_cross_row, quad_info.cross.row)
    else:
        if len(quad_info.backward.left) > 0:
            if len(search_start_stack) == 0 or search_start_stack[-1].index != quad_info.backward.left[0].index:
                search_start_stack.append(quad_info.backward.left[0])

        quad_info.cross.row = max(1, quad_info.cross.row)

    return quad_info.cross.row


def reverse_deep_search_col_cross(quad_info: QuadInfo, search_start_stack: List[QuadInfo]) -> int:
    """
    反向深度搜索四边形跨列信息

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    if len(quad_info.forward.top) > 0:
        total_cross_col = 0

        for next_quad_info in quad_info.forward.top:
            total_cross_col += reverse_deep_search_col_cross(next_quad_info, search_start_stack)

        quad_info.cross.col = max(total_cross_col, quad_info.cross.col)
    else:
        if len(quad_info.backward.top) > 0:
            if len(search_start_stack) == 0 or search_start_stack[-1].index != quad_info.backward.top[0].index:
                search_start_stack.append(quad_info.backward.top[0])

        quad_info.cross.col = max(1, quad_info.cross.col)

    return quad_info.cross.col


def deep_search_row_range(quad_info: QuadInfo, row_range_start: int, search_start_stack: List[Tuple[QuadInfo, int]]) -> int:
    """
    正向深度搜索四边形跨行信息并计算每个四边形（单元格）行范围

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    quad_info.row_range.start = row_range_start

    if len(quad_info.forward.right) > 0:
        total_cross_row = 0

        for next_quad_info in quad_info.forward.right:
            total_cross_row += deep_search_row_range(next_quad_info, row_range_start + total_cross_row, search_start_stack)

        quad_info.cross.row = max(total_cross_row, quad_info.cross.row)
    else:
        if len(quad_info.backward.right) > 0:
            if len(search_start_stack) == 0 or search_start_stack[-1][0].index != quad_info.backward.right[0].index:
                search_start_stack.append((quad_info.backward.right[0], row_range_start))

        quad_info.cross.row = max(1, quad_info.cross.row)

    quad_info.row_range.end = quad_info.row_range.start + quad_info.cross.row - 1

    return quad_info.cross.row


def deep_search_col_range(quad_info: QuadInfo, col_range_start: int, search_start_stack: List[Tuple[QuadInfo, int]]) -> int:
    """
    正向深度搜索四边形跨列信息并计算每个四边形（单元格）列范围

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    quad_info.col_range.start = col_range_start

    if len(quad_info.forward.bottom) > 0:
        total_cross_col = 0

        for next_quad_info in quad_info.forward.bottom:
            total_cross_col += deep_search_col_range(next_quad_info, col_range_start + total_cross_col, search_start_stack)

        quad_info.cross.col = max(total_cross_col, quad_info.cross.col)
    else:
        if len(quad_info.backward.bottom) > 0:
            if len(search_start_stack) == 0 or search_start_stack[-1][0].index != quad_info.backward.bottom[0].index:
                search_start_stack.append((quad_info.backward.bottom[0], col_range_start))

        quad_info.cross.col = max(1, quad_info.cross.col)

    quad_info.col_range.end = quad_info.col_range.start + quad_info.cross.col - 1

    return quad_info.cross.col


def split_quads(quads_info: List[QuadInfo]) -> List[List[QuadInfo]]:
    """
    将多表格的四边形列表进行分割，得到多表格四边形列表

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    # 还原索引
    quads_info.sort(key=lambda quad_info: quad_info.index)

    # 构建并查集
    disjoint_set = DisjointSet(quads_info)

    # 合并集合
    for quad_info1 in quads_info:
        # 左侧
        for quad_info2 in quad_info1.forward.left:
            disjoint_set.union(quad_info1.index, quad_info2.index)
        # 右侧
        for quad_info2 in quad_info1.forward.right:
            disjoint_set.union(quad_info1.index, quad_info2.index)
        # 上侧
        for quad_info2 in quad_info1.forward.top:
            disjoint_set.union(quad_info1.index, quad_info2.index)
        # 下侧
        for quad_info2 in quad_info1.forward.bottom:
            disjoint_set.union(quad_info1.index, quad_info2.index)

    return disjoint_set.get_all_collections()


def update_quads_cross(quads_info: List[QuadInfo], search_mode: Literal["forward", "backward"], search_target: Literal["row", "col", "all"]) -> None:
    """
    更新四边形（单元格）跨行信息

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    if search_mode == "forward":  ###* 正向搜索
        # 查找到左上角四边形框
        left_top_quad_info = min(quads_info, key=lambda quad_info: quad_info.quad[0].x + quad_info.quad[0].y)

        # 正向搜索起点栈
        search_start_stack: List[QuadInfo]

        if search_target == "row" or search_target == "all":
            # 重置正向搜索起点栈
            search_start_stack = [left_top_quad_info]

            # 遍历整个栈，更新行结构
            while len(search_start_stack) > 0:
                search_quad_info = search_start_stack.pop()

                # 更新每个四边形的行跨度
                deep_search_row_cross(search_quad_info, search_start_stack)

                if len(search_quad_info.forward.bottom) > 0:
                    search_start_stack.append(search_quad_info.forward.bottom[0])
                elif len(search_quad_info.backward.bottom) > 0:
                    search_start_stack.append(search_quad_info.backward.bottom[0])

        if search_target == "col" or search_target == "all":
            # 重置正向搜索起点栈
            search_start_stack = [left_top_quad_info]

            # 遍历整个栈，更新列结构
            while len(search_start_stack) > 0:
                search_quad_info = search_start_stack.pop()

                # 更新每个四边形的列跨度
                deep_search_col_cross(search_quad_info, search_start_stack)

                if len(search_quad_info.forward.right) > 0:
                    search_start_stack.append(search_quad_info.forward.right[0])
                elif len(search_quad_info.backward.right) > 0:
                    search_start_stack.append(search_quad_info.backward.right[0])
    else:  ###* 反向搜索
        # 查找到右下角四边形框
        right_bottom_quad_info = max(quads_info, key=lambda quad_info: quad_info.quad[2].x + quad_info.quad[2].y)

        # 反向搜索起点栈
        reverse_search_start_stack: List[QuadInfo]

        if search_target == "row" or search_target == "all":
            # 重置反向搜索起点栈
            reverse_search_start_stack = [right_bottom_quad_info]

            # 遍历整个栈，更新跨行信息
            while len(reverse_search_start_stack) > 0:
                search_quad_info = reverse_search_start_stack.pop()

                # 更新每个四边形的行跨度
                reverse_deep_search_row_cross(search_quad_info, reverse_search_start_stack)

                if len(search_quad_info.forward.top) > 0:
                    reverse_search_start_stack.append(search_quad_info.forward.top[0])
                elif len(search_quad_info.backward.top) > 0:
                    reverse_search_start_stack.append(search_quad_info.backward.top[0])

        if search_target == "col" or search_target == "all":
            # 重置反向搜索起点栈
            reverse_search_start_stack = [right_bottom_quad_info]

            # 遍历整个栈，更新跨列信息
            while len(reverse_search_start_stack) > 0:
                search_quad_info = reverse_search_start_stack.pop()

                # 更新每个四边形的列跨度
                reverse_deep_search_col_cross(search_quad_info, reverse_search_start_stack)

                if len(search_quad_info.forward.left) > 0:
                    reverse_search_start_stack.append(search_quad_info.forward.left[0])
                elif len(search_quad_info.backward.left) > 0:
                    reverse_search_start_stack.append(search_quad_info.backward.left[0])


def adjust_special_quads_cross(quads_info: List[QuadInfo]) -> None:
    """
    调整特殊四边形（单元格）的跨行跨列信息

    Autor: dreamy-xay
    Date: 2024-07-09
    """

    # 循环查找下一个相邻的长度不为1的quads_info
    def find_next_quads_info(quads_info_temp: List[QuadInfo], dir: str) -> List[QuadInfo]:
        while len(quads_info_temp) == 1:
            quads_info_temp = getattr(quads_info_temp[0].forward, dir)

        return quads_info_temp

    # 深度搜索所有跨列数小的子项
    def deep_search(quads_info_temp: List[QuadInfo], dir: str) -> List[QuadInfo]:
        new_quads_info_temp: List[QuadInfo] = []
        for quad_info_temp in quads_info_temp:
            if len(quad_info_temp.cache) > 1:
                new_quads_info_temp.extend(quad_info_temp.cache)
                continue

            dir_quads_info = deep_search(getattr(quad_info_temp.forward, dir), dir)

            if len(dir_quads_info) <= 1:
                new_quads_info_temp.append(quad_info_temp)
            else:
                new_quads_info_temp.extend(dir_quads_info)

        return new_quads_info_temp

    # 运行函数入口
    def run(adjust_target: Literal["col", "row"]) -> None:
        if adjust_target == "col":
            dir1, dir2 = ("top", "bottom")
            start_cmp_index = (0, 3)
            end_cmp_index = (1, 2)
        else:
            dir1, dir2 = ("left", "right")
            start_cmp_index = (1, 0)
            end_cmp_index = (2, 3)

        # 先按照上下相邻四边形的数目升序排序
        quads_info.sort(key=lambda quad_info: max(len(getattr(quad_info.forward, dir1)), len(getattr(quad_info.forward, dir2))), reverse=True)

        # 记录已执行过的数据索引集合
        indexes_set: Set[int] = set()

        # 深度搜索模式
        search_mode: Literal["forward", "backward"]

        # 遍历每个相邻四边形，更新相邻四边形的结构
        for quad_info in quads_info:
            # 获取相邻的四边形
            dir1_quads_info = getattr(quad_info.forward, dir1)
            dir2_quads_info = getattr(quad_info.forward, dir2)

            # 计算相邻四边形的数目
            temp_quads_len = len(dir1_quads_info)
            ref_quads_len = len(dir2_quads_info)

            # 如果任何一个方向不存在相邻四边形，则直接跳过
            if temp_quads_len <= 0 or ref_quads_len <= 0:
                continue

            # 迭代查询间接相邻的四边形
            quads_info_temp, quads_info_ref = find_next_quads_info(dir1_quads_info, dir1), find_next_quads_info(dir2_quads_info, dir2)

            # 计算间接相邻四边形的数目
            temp_quads_len = len(quads_info_temp)
            ref_quads_len = len(quads_info_ref)

            # 如果相同数据已经执行过了，则直接跳过
            if sum([int(quad_info.index in indexes_set) for quad_info in quads_info_temp]) == temp_quads_len:
                continue

            # 加入已执行过的数据索引集合
            indexes_set.update([quad_info.index for quad_info in quads_info_temp])
            indexes_set.update([quad_info.index for quad_info in quads_info_ref])

            # 如果任何一个方向的间接相邻四边形数目小于等于1，则直接跳过
            if temp_quads_len <= 1 or ref_quads_len <= 1:
                continue

            # 深度搜索所有跨列数小的子项
            quads_info_temp_extend, quads_info_ref_extend = deep_search(quads_info_temp, dir1), deep_search(quads_info_ref, dir2)

            # 计算扩展后的间接相邻四边形的数目
            temp_quads_extend_len = len(quads_info_temp_extend)
            ref_quads_extend_len = len(quads_info_ref_extend)

            # 需要处理的数据列表
            process_data_list: List[Tuple[List[QuadInfo], List[QuadInfo], Literal["forward", "backward"], List[QuadInfo]]] = []

            # 根据不同状态处理不同的数据
            if temp_quads_extend_len < ref_quads_extend_len:
                process_data_list.append((quads_info_temp, quads_info_ref_extend, "forward", quads_info_ref))
            elif temp_quads_extend_len > ref_quads_extend_len:
                process_data_list.append((quads_info_ref, quads_info_temp_extend, "backward", quads_info_temp))
            else:
                if temp_quads_len != temp_quads_extend_len:
                    process_data_list.append((quads_info_temp, quads_info_ref_extend, "forward", quads_info_ref))
                if ref_quads_len != ref_quads_extend_len:
                    process_data_list.append((quads_info_ref, quads_info_temp_extend, "backward", quads_info_temp))

            # 循环处理数据
            for quads_info_temp, quads_info_ref, search_mode, ori_quads_info_ref in process_data_list:
                # print("-------------------------:", [quad_info.index for quad_info in quads_info_temp], [quad_info.index for quad_info in ori_quads_info_ref])
                # print("+++++++++++++++++++++++++:", [quad_info.index for quad_info in quads_info_temp], [quad_info.index for quad_info in quads_info_ref])

                quads_info_ref_indexes = [i for i in range(len(quads_info_ref))]

                # 调整每个特殊四边形（单元格）的跨行跨列信息
                for quad_info_temp in quads_info_temp:
                    # 获取跨行跨列索引信息
                    index1 = min(
                        quads_info_ref_indexes,
                        key=lambda i: quads_info_ref[i]
                        .quad[start_cmp_index[0]]
                        .middle(quads_info_ref[i].quad[start_cmp_index[1]])
                        .distance(quad_info_temp.quad[start_cmp_index[0]], quad_info_temp.quad[start_cmp_index[1]]),
                    )
                    index2 = min(
                        quads_info_ref_indexes,
                        key=lambda i: quads_info_ref[i]
                        .quad[end_cmp_index[0]]
                        .middle(quads_info_ref[i].quad[end_cmp_index[1]])
                        .distance(quad_info_temp.quad[end_cmp_index[0]], quad_info_temp.quad[end_cmp_index[1]]),
                    )

                    # 如果索引正常
                    if index1 <= index2:
                        quad_info_temp.cache.clear()  # 清除缓存
                        quad_info_temp.cache.extend(quads_info_ref[index1 : index2 + 1])  # 更新缓存
                        new_cross = sum(map(lambda quad_info: getattr(quad_info.cross, adjust_target), quads_info_ref[index1 : index2 + 1]))

                        # print("=>>>", getattr(quad_info_temp.cross, adjust_target), new_cross)

                        # 如果新的跨行跨列更大，则更新
                        if getattr(quad_info_temp.cross, adjust_target) < new_cross:
                            setattr(quad_info_temp.cross, adjust_target, new_cross)
                            update_quads_cross(quads_info, search_mode, adjust_target)

    # 调整特殊四边形（单元格）的跨列信息
    run("col")
    # 调整特殊四边形（单元格）的跨行信息
    run("row")


def update_quads_structure(quads_info: List[QuadInfo]) -> None:
    """
    根据关系图更新每个四边形（单元格）的结构

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    ###* 正向搜索四边形（单元格）的跨行跨列信息
    update_quads_cross(quads_info, "forward", "all")

    ###* 反向搜索四边形（单元格）的跨行跨列信息
    update_quads_cross(quads_info, "backward", "all")

    ###* 调整特殊四边形（单元格）的跨行跨列信息
    adjust_special_quads_cross(quads_info)

    ###* 正向搜索并更新四边形（单元格）的行列结构
    # 查找到左上角四边形框
    left_top_quad_info = min(quads_info, key=lambda quad_info: quad_info.quad[0].x + quad_info.quad[0].y)

    # 搜索起点栈
    search_start_stack: List[Tuple[QuadInfo, int]] = [(left_top_quad_info, 0)]

    # 遍历整个栈，更新行结构
    while len(search_start_stack) > 0:
        search_quad_info, row_range_start = search_start_stack.pop()

        # 当前跨度
        cross_row = deep_search_row_range(search_quad_info, row_range_start, search_start_stack)

        if len(search_quad_info.forward.bottom) > 0:
            search_start_stack.append((search_quad_info.forward.bottom[0], row_range_start + cross_row))
        elif len(search_quad_info.backward.bottom) > 0:
            search_start_stack.append((search_quad_info.backward.bottom[0], row_range_start + cross_row))

    # 重置搜索起点栈
    search_start_stack = [(left_top_quad_info, 0)]

    # 遍历整个栈，更新列结构
    while len(search_start_stack) > 0:
        search_quad_info, col_range_start = search_start_stack.pop()

        # 当前跨度
        cross_col = deep_search_col_range(search_quad_info, col_range_start, search_start_stack)

        if len(search_quad_info.forward.right) > 0:
            search_start_stack.append((search_quad_info.forward.right[0], col_range_start + cross_col))
        elif len(search_quad_info.backward.right) > 0:
            search_start_stack.append((search_quad_info.backward.right[0], col_range_start + cross_col))


def get_quads_structure(quads: List[List[Tuple[int, int]]], adjacent_threshold: float = 10) -> List[List[QuadInfo]]:
    """
    获取四边形列表的结构信息

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    ###* 构建关系图
    quads_info = build_relation_map(quads, adjacent_threshold, True)

    ###* 分割多表格
    quads_info_list = split_quads(quads_info)

    ###* 根据关系图更新每个四边形（单元格）的结构
    for quads_info_item in quads_info_list:
        update_quads_structure(quads_info_item)

    return quads_info_list
