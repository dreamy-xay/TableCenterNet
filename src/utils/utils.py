#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 14:49:10
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 14:49:13
"""
import numpy as np
from bisect import bisect_left, bisect_right


class IntervalFinder:
    def __init__(self, intervals):
        """
        初始化 IntervalFinder 类，接受一个由 (start, end) 区间组成的列表作为输入。

        :param intervals: 一个包含多个 (start, end) 区间元组的列表
        """
        # 记录区间数目
        num_intervals = len(intervals)

        # 对区间按照起始点进行排序，并保留原始索引
        self.start_sorted = sorted(list(range(num_intervals)), key=lambda idx: intervals[idx][0])
        # 对区间按照结束点进行排序，并保留原始索引
        self.end_sorted = sorted(list(range(num_intervals)), key=lambda idx: intervals[idx][1])

        # 提取出仅用于二分查找的起始点和结束点的列表
        self.starts = [intervals[idx][0] for idx in self.start_sorted]
        self.ends = [intervals[idx][1] for idx in self.end_sorted]

    def query(self, q_start, q_end, is_union=True):
        """
        查询与给定区间相交的所有区间索引。

        :param q_start: 查询区间的起始点
        :param q_end: 查询区间的结束点
        :param is_union: 是否求并集，否则求交集
        :return: 一个包含所有相交区间原始索引的列表
        """
        # 找出所有起始点在查询区间内的区间索引
        start_index = bisect_left(self.starts, q_start)
        end_index = bisect_right(self.starts, q_end)
        start_indexes = set(self.start_sorted[start_index:end_index])

        # 找出所有结束点在查询区间内的区间索引
        start_index = bisect_left(self.ends, q_start)
        end_index = bisect_right(self.ends, q_end)
        end_indexes = set(self.end_sorted[start_index:end_index])

        # 取并集
        return start_indexes.union(end_indexes) if is_union else start_indexes.intersection(end_indexes)


class BoxesFinder:
    def __init__(self, x_ranges, y_ranges, query_x_ranges, query_y_ranges):
        """
        快速查询 boxes 中哪些 box 与 query_boxes 中指定的 query_box 相交的类

        :param x_ranges: boxes 中每个 box 的 x 坐标范围
        :param y_ranges: boxes 中每个 box 的 y 坐标范围
        :param query_x_ranges: query_boxes 中每个 query_box 的 x 坐标范围
        :param query_y_ranges: query_boxes 中每个 query_box 的 y 坐标范围

        """
        # 初始化 query
        self.query_x_intervals = query_x_ranges
        self.query_y_intervals = query_y_ranges

        # 构建区间查询器，查询 boxes 中哪些 box 与 query_boxes 中指定的 query_box 相交
        self.x_interval_finder = IntervalFinder(x_ranges)
        self.y_interval_finder = IntervalFinder(y_ranges)

        # 获取 query_interval 被 interval 包含的情况
        self.included_x_indexes_list = self.get_included(query_x_ranges, x_ranges)
        self.included_y_indexes_list = self.get_included(query_y_ranges, y_ranges)

    @staticmethod
    def get_included(query_intervals, intervals):
        query_interval_finder = IntervalFinder(query_intervals)

        indexes_list = [set() for _ in range(len(query_intervals))]
        for i, interval in enumerate(intervals):
            query_interval_indexes = query_interval_finder.query(*interval, False)
            for query_interval_index in query_interval_indexes:
                indexes_list[query_interval_index].add(i)

        return indexes_list

    def query(self, index):
        """
        快速查询 boxes 中哪些 box 与 query_boxes 中指定索引的 query_box 相交
        query 时间复杂度 O(max(log(n), m)), n 表示 boxes 数目，m 表示与 query_box 相交的 box 的最大数目

        :param index: 索引
        :return: 与 query_boxes 中指定索引的 query_box 相交的 box 的索引集合
        """
        # 获取查询区间
        x_interval = self.query_x_intervals[index]
        y_interval = self.query_y_intervals[index]

        # 获取包含查询区间（query_intervals）的区间（intervals）索引集合
        included_x_indexes = self.included_x_indexes_list[index]
        included_y_indexes = self.included_y_indexes_list[index]

        # 获取与查询区间（query_intervals）相交的区间（intervals）索引集合
        x_indexes = self.x_interval_finder.query(*x_interval).union(included_x_indexes)
        y_indexes = self.y_interval_finder.query(*y_interval).union(included_y_indexes)

        return x_indexes.intersection(y_indexes)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        if self.val != 0:
            self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        # 如果是 numpy 数组，转换为 list
        return obj.tolist()
    elif isinstance(obj, dict):
        # 如果是字典，递归地将键值对中的 numpy 数组转换为 list
        return {key: numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # 如果是列表，递归地处理每个元素
        return [numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        # 如果是元组，递归地处理每个元素
        return tuple(numpy_to_list(item) for item in obj)
    else:
        # 如果既不是 numpy 数组，也不是容器类型，直接返回原始对象
        return obj
