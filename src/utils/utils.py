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
        Initialize the IntervalFinder class to accept a list of (start, end) intervals as input.

        :param intervals: A list of multiple (start, end) interval tuples
        """
        # Number of record intervals
        num_intervals = len(intervals)

        # Sort the intervals by starting point and keep the original index
        self.start_sorted = sorted(list(range(num_intervals)), key=lambda idx: intervals[idx][0])
        # Sort intervals by end point and retain the original index
        self.end_sorted = sorted(list(range(num_intervals)), key=lambda idx: intervals[idx][1])

        # Extract a list of start and end points that are only used for binary lookups
        self.starts = [intervals[idx][0] for idx in self.start_sorted]
        self.ends = [intervals[idx][1] for idx in self.end_sorted]

    def query(self, q_start, q_end, is_union=True):
        """
        Query all compartment indexes that intersect a given compartment.

        :param q_start: The starting point of the query interval
        :param q_end: The end point of the query compartment
        :param is_union: Whether to find the union, otherwise to find the intersection
        :return: A list of the original indexes of all intersecting intervals
        """
        # Find the interval index of all starting points within the query interval
        start_index = bisect_left(self.starts, q_start)
        end_index = bisect_right(self.starts, q_end)
        start_indexes = set(self.start_sorted[start_index:end_index])

        # Find the interval index of all end points within the query interval
        start_index = bisect_left(self.ends, q_start)
        end_index = bisect_right(self.ends, q_end)
        end_indexes = set(self.end_sorted[start_index:end_index])

        # Take the union
        return start_indexes.union(end_indexes) if is_union else start_indexes.intersection(end_indexes)


class BoxesFinder:
    def __init__(self, x_ranges, y_ranges, query_x_ranges, query_y_ranges):
        """
        Quickly query which boxes in the boxes intersect with the query_box specified in query_boxes

        :param x_ranges: The range of x-coordinates for each box in the boxes
        :param y_ranges: The y-coordinate range of each box in the boxes
        :param query_x_ranges: The x-coordinate range for each query_box in the query_boxes
        :param query_y_ranges: The y-coordinate range for each query_box in the query_boxes

        """
        # Initialize the query
        self.query_x_intervals = query_x_ranges
        self.query_y_intervals = query_y_ranges

        # Build an interval querier to query which boxes in the boxes intersect with the query_box specified in the query_boxes
        self.x_interval_finder = IntervalFinder(x_ranges)
        self.y_interval_finder = IntervalFinder(y_ranges)

        # Get the case where query_interval is included in interval
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
        Quickly query which boxes in the boxes intersect with the query_box of the specified index in query_boxes
        The query time complexity O(max(log(n), m)), n is the number of boxes, and m is the maximum number of boxes that intersect query_box

        :param index: index
        :return: The collection of indexes for boxes that intersect the query_box of the specified index in query_boxes
        """
        # Obtain the query interval
        x_interval = self.query_x_intervals[index]
        y_interval = self.query_y_intervals[index]

        # Get the set of intervals indexes that contain the query_intervals of query intervals
        included_x_indexes = self.included_x_indexes_list[index]
        included_y_indexes = self.included_y_indexes_list[index]

        # Get the set of intervals indexes that intersect the query interval (query_intervals).
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
        # If it's a numpy array, convert it to a list
        return obj.tolist()
    elif isinstance(obj, dict):
        # If it's a dictionary, recursively convert the numpy array in the key-value pair to a list
        return {key: numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # In the case of a list, each element is processed recursively
        return [numpy_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        # In the case of tuples, each element is processed recursively
        return tuple(numpy_to_list(item) for item in obj)
    else:
        # If it's neither a numpy array nor a container type, just return the original object
        return obj
