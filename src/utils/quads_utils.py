#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: Get the cell structure
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


# * Interval objects
@dataclass
class Interval:
    start: int  # Start of the interval
    end: int  # End of interval


# * Cross-row and cross-column information
@dataclass
class CrossInfo:
    row: int  # Inter-bank information
    col: int  # Cross-column information


# * Relationship diagram
@dataclass
class RelationMap:
    top: List["QuadInfo"]
    right: List["QuadInfo"]
    bottom: List["QuadInfo"]
    left: List["QuadInfo"]


# * Quadrilateral (cell) information
@dataclass
class QuadInfo:
    # Data
    quad: List["Point"]

    # Index information
    index: int

    # Structural information
    row_range: Interval
    col_range: Interval
    cross: CrossInfo

    # Diagram Information
    forward: RelationMap  # Forward (Inclusion or Equivalent) Diagrams
    backward: RelationMap  # Reverse (Included) Diagram

    # Cache information
    cache: List["QuadInfo"]


# * Point object
class Point:
    def __init__(self, point: Tuple[int, int]) -> None:
        self.x = point[0]
        self.y = point[1]

    def is_adjacent(self, other: "Point", threshold: int = 10) -> bool:
        """
        Determine if two points are next to each other

        Autor: dreamy-xay
        Date: 2024-07-09
        """
        return abs(self.x - other.x) < threshold and abs(self.y - other.y) < threshold

    def distance(self, line_point1: "Point", line_point2: "Point") -> float:
        """
        Calculates the distance from a point to a straight line

        Autor: dreamy-xay
        Date: 2024-07-09
        """
        # Calculate the general equation of the straight line
        # Ax   By   C = 0
        A = line_point2.y - line_point1.y
        B = line_point1.x - line_point2.x
        C = line_point2.x * line_point1.y - line_point1.x * line_point2.y

        # Calculate point-to-straight distance
        # d = | Ax   By   C| / sqrt(A^2   B^2)
        return abs(A * self.x + B * self.y + C) / math.sqrt(A**2 + B**2)

    def middle(self, other: "Point") -> "Point":
        """
        Calculate the midpoint of two points

        Autor: dreamy-xay
        Date: 2024-07-09
        """
        return Point(((self.x + other.x) // 2, (self.y + other.y) // 2))


# And check the set
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
        Merge two elements into a single collection

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
        Get all collections (after the merge is complete).

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
    Finds a list of all contiguous quads (including start and end quads) from one start quad to the middle of another end quad

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    # Initialize the data
    temp_quads_info: List[QuadInfo] = [start_quad_info]
    temp_quads_keys = set()

    # Brute-force iteration query for a list of all consecutive quads from the start quad to the middle of the end quad
    while True:
        temp_quads_keys.add(temp_quads_info[-1].index)
        temp_quad_info: Optional[QuadInfo] = None

        # Find the next quad
        next_index = next_cache[temp_quads_info[-1].index]
        if next_index != -1:
            temp_quad_info = quads_info[next_index]

        if temp_quad_info is None or temp_quad_info.index == end_quad_info.index:
            break

        # Infinite Loop, just quit
        if temp_quad_info.index in temp_quads_keys:
            temp_quads_info = [start_quad_info]
            break

        # Add to the list
        temp_quads_info.append(temp_quad_info)

    # Add a termination quad
    temp_quads_info.append(end_quad_info)

    # The query is completed and the result is returned
    return temp_quads_info


def build_sub_relation_map(quads_info: List[QuadInfo], dir: Literal["left", "right", "top", "bottom"], adjacent_threshold: float, strict_mode: bool = True) -> None:
    """
    Construct one-way forward and backward diagrams

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    # Initialize the data
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

    # Quad list length
    num_quads = len(quads_info)

    # Check out the initialization data
    start_index1, start_index2 = start_point_index_pair
    end_index1, end_index2 = end_point_index_pair

    # Create a next cache
    next_cache = None

    def build_next_cache(next_cache):  # next Cache is cached only once, only when needed
        if next_cache is not None:
            return next_cache
        next_cache = [-1] * num_quads
        for i in range(num_quads):
            for j in range(num_quads):
                if i != j and quads_info[i].quad[end_index2].is_adjacent(quads_info[j].quad[start_index2], adjacent_threshold):
                    next_cache[quads_info[i].index] = j
                    break
        return next_cache

    ### Construction of one-way diagrams
    for i in range(num_quads):
        quad_info = quads_info[i]

        # Find the start rectangle
        start_quad_info: Optional[QuadInfo] = None
        for j in range(num_quads):
            if i != j and quad_info.quad[start_index1].is_adjacent(quads_info[j].quad[start_index2], adjacent_threshold):
                start_quad_info = quads_info[j]
                break

        # Find the end rectangle
        end_quad_info: Optional[QuadInfo] = None
        for j in range(num_quads):
            if i != j and quad_info.quad[end_index1].is_adjacent(quads_info[j].quad[end_index2], adjacent_threshold):
                end_quad_info = quads_info[j]

        # Build a diagram
        if (start_quad_info is not None) and (end_quad_info is not None):
            dir_relation_map: List[QuadInfo] = getattr(quad_info.forward, dir)
            if start_quad_info.index == end_quad_info.index:
                dir_relation_map.append(start_quad_info)
            else:
                next_cache = build_next_cache(next_cache)  # It will only be created once
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
    Construct forward and backward diagrams in four directions

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    num_quads = len(quads)

    # Initialize the quad list information
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

    # Sort by the Y coordinates in the upper left corner
    quads_info.sort(key=lambda quad_info: quad_info.quad[0].y)

    # Construction of Left Diagram
    build_sub_relation_map(quads_info, "left", adjacent_threshold, strict_mode)
    # Construction of the right diagram
    build_sub_relation_map(quads_info, "right", adjacent_threshold, strict_mode)

    # Sort by the x coordinate in the upper left corner
    quads_info.sort(key=lambda quad_info: quad_info.quad[0].x)

    # Construction of the relationship diagram
    build_sub_relation_map(quads_info, "top", adjacent_threshold, strict_mode)
    # Construction of the next diagram
    build_sub_relation_map(quads_info, "bottom", adjacent_threshold, strict_mode)

    return quads_info


def deep_search_row_cross(quad_info: QuadInfo, search_start_stack: List[QuadInfo]) -> int:
    """
    Forward deep search quad cross-row information

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
    Forward depth search quad cross-column information

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
    Reverse deep search quadrilateral cross-row information

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
    Reverse deep search quadrilateral cross-column information

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
    Forward depth searches for quad spanning information and calculates the row range for each quad (cell).

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
    Forward depth searches quad across column information and calculates the column range for each quadrilateral (cell).

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
    Split the quadrilateral list of the multitable to obtain the multi-table quadrilateral list

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    # Restore indexes
    quads_info.sort(key=lambda quad_info: quad_info.index)

    # Build and look up the set
    disjoint_set = DisjointSet(quads_info)

    # Merge Collections
    for quad_info1 in quads_info:
        # Left side
        for quad_info2 in quad_info1.forward.left:
            disjoint_set.union(quad_info1.index, quad_info2.index)
        # Right side
        for quad_info2 in quad_info1.forward.right:
            disjoint_set.union(quad_info1.index, quad_info2.index)
        # Upper side
        for quad_info2 in quad_info1.forward.top:
            disjoint_set.union(quad_info1.index, quad_info2.index)
        # Underside
        for quad_info2 in quad_info1.forward.bottom:
            disjoint_set.union(quad_info1.index, quad_info2.index)

    return disjoint_set.get_all_collections()


def update_quads_cross(quads_info: List[QuadInfo], search_mode: Literal["forward", "backward"], search_target: Literal["row", "col", "all"]) -> None:
    """
    Update quad (cell) cross-row information

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    if search_mode == "forward":  ###* Forward search
        # Find the quadrilateral box in the upper left corner
        left_top_quad_info = min(quads_info, key=lambda quad_info: quad_info.quad[0].x + quad_info.quad[0].y)

        # Forward search for the starting stack
        search_start_stack: List[QuadInfo]

        if search_target == "row" or search_target == "all":
            # Reset the forward search starting stack
            search_start_stack = [left_top_quad_info]

            # Traverse the entire stack and update the row structure
            while len(search_start_stack) > 0:
                search_quad_info = search_start_stack.pop()

                # Update the row span of each quad
                deep_search_row_cross(search_quad_info, search_start_stack)

                if len(search_quad_info.forward.bottom) > 0:
                    search_start_stack.append(search_quad_info.forward.bottom[0])
                elif len(search_quad_info.backward.bottom) > 0:
                    search_start_stack.append(search_quad_info.backward.bottom[0])

        if search_target == "col" or search_target == "all":
            # Reset the forward search starting stack
            search_start_stack = [left_top_quad_info]

            # Traverse the entire stack and update the column structure
            while len(search_start_stack) > 0:
                search_quad_info = search_start_stack.pop()

                # Update the column span for each quad
                deep_search_col_cross(search_quad_info, search_start_stack)

                if len(search_quad_info.forward.right) > 0:
                    search_start_stack.append(search_quad_info.forward.right[0])
                elif len(search_quad_info.backward.right) > 0:
                    search_start_stack.append(search_quad_info.backward.right[0])
    else:  ###* Reverse search
        # Find the quad box in the lower right corner
        right_bottom_quad_info = max(quads_info, key=lambda quad_info: quad_info.quad[2].x + quad_info.quad[2].y)

        # Reverse search for the starting stack
        reverse_search_start_stack: List[QuadInfo]

        if search_target == "row" or search_target == "all":
            # Reset the reverse search starting stack
            reverse_search_start_stack = [right_bottom_quad_info]

            # Traverse the entire stack and update the inter-row information
            while len(reverse_search_start_stack) > 0:
                search_quad_info = reverse_search_start_stack.pop()

                # Update the row span of each quad
                reverse_deep_search_row_cross(search_quad_info, reverse_search_start_stack)

                if len(search_quad_info.forward.top) > 0:
                    reverse_search_start_stack.append(search_quad_info.forward.top[0])
                elif len(search_quad_info.backward.top) > 0:
                    reverse_search_start_stack.append(search_quad_info.backward.top[0])

        if search_target == "col" or search_target == "all":
            # Reset the reverse search starting stack
            reverse_search_start_stack = [right_bottom_quad_info]

            # Traverse the entire stack and update the cross-column information
            while len(reverse_search_start_stack) > 0:
                search_quad_info = reverse_search_start_stack.pop()

                # Update the column span for each quad
                reverse_deep_search_col_cross(search_quad_info, reverse_search_start_stack)

                if len(search_quad_info.forward.left) > 0:
                    reverse_search_start_stack.append(search_quad_info.forward.left[0])
                elif len(search_quad_info.backward.left) > 0:
                    reverse_search_start_stack.append(search_quad_info.backward.left[0])


def adjust_special_quads_cross(quads_info: List[QuadInfo]) -> None:
    """
    Adjust the cross-row and cross-column information of special quads (cells).

    Autor: dreamy-xay
    Date: 2024-07-09
    """

    # Loop through the next adjacent quads_info that is not 1 in length
    def find_next_quads_info(quads_info_temp: List[QuadInfo], dir: str) -> List[QuadInfo]:
        while len(quads_info_temp) == 1:
            quads_info_temp = getattr(quads_info_temp[0].forward, dir)

        return quads_info_temp

    # Deep search for all children with a small number of spanning columns
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

    # Run the function entry
    def run(adjust_target: Literal["col", "row"]) -> None:
        if adjust_target == "col":
            dir1, dir2 = ("top", "bottom")
            start_cmp_index = (0, 3)
            end_cmp_index = (1, 2)
        else:
            dir1, dir2 = ("left", "right")
            start_cmp_index = (1, 0)
            end_cmp_index = (2, 3)

        # Sort in ascending order by the number of adjacent quadrilaterals above and below
        quads_info.sort(key=lambda quad_info: max(len(getattr(quad_info.forward, dir1)), len(getattr(quad_info.forward, dir2))), reverse=True)

        # Record the set of data indexes that have been executed
        indexes_set: Set[int] = set()

        # Deep search mode
        search_mode: Literal["forward", "backward"]

        # Iterate through each adjacent quad to update the structure of the adjacent quadrilateral
        for quad_info in quads_info:
            # Get adjacent quads
            dir1_quads_info = getattr(quad_info.forward, dir1)
            dir2_quads_info = getattr(quad_info.forward, dir2)

            # Count the number of adjacent quads
            temp_quads_len = len(dir1_quads_info)
            ref_quads_len = len(dir2_quads_info)

            # If there are no adjacent quads in either direction, skip them directly
            if temp_quads_len <= 0 or ref_quads_len <= 0:
                continue

            # Iteratively query indirectly adjacent quads
            quads_info_temp, quads_info_ref = find_next_quads_info(dir1_quads_info, dir1), find_next_quads_info(dir2_quads_info, dir2)

            # Count the number of indirect adjacent quads
            temp_quads_len = len(quads_info_temp)
            ref_quads_len = len(quads_info_ref)

            # If the same data has already been executed, skip it
            if sum([int(quad_info.index in indexes_set) for quad_info in quads_info_temp]) == temp_quads_len:
                continue

            # Add a collection of executed data indexes
            indexes_set.update([quad_info.index for quad_info in quads_info_temp])
            indexes_set.update([quad_info.index for quad_info in quads_info_ref])

            # If the number of indirect adjacent quadrilaterals in any one direction is less than or equal to 1, skip directly
            if temp_quads_len <= 1 or ref_quads_len <= 1:
                continue

            # Deep search for all children with a small number of spanning columns
            quads_info_temp_extend, quads_info_ref_extend = deep_search(quads_info_temp, dir1), deep_search(quads_info_ref, dir2)

            # Count the number of indirect adjacent quads after the expansion
            temp_quads_extend_len = len(quads_info_temp_extend)
            ref_quads_extend_len = len(quads_info_ref_extend)

            # List of data to be processed
            process_data_list: List[Tuple[List[QuadInfo], List[QuadInfo], Literal["forward", "backward"], List[QuadInfo]]] = []

            # Handle different data according to different statuses
            if temp_quads_extend_len < ref_quads_extend_len:
                process_data_list.append((quads_info_temp, quads_info_ref_extend, "forward", quads_info_ref))
            elif temp_quads_extend_len > ref_quads_extend_len:
                process_data_list.append((quads_info_ref, quads_info_temp_extend, "backward", quads_info_temp))
            else:
                if temp_quads_len != temp_quads_extend_len:
                    process_data_list.append((quads_info_temp, quads_info_ref_extend, "forward", quads_info_ref))
                if ref_quads_len != ref_quads_extend_len:
                    process_data_list.append((quads_info_ref, quads_info_temp_extend, "backward", quads_info_temp))

            # Loop processing data
            for quads_info_temp, quads_info_ref, search_mode, ori_quads_info_ref in process_data_list:
                # print("-------------------------:", [quad_info.index for quad_info in quads_info_temp], [quad_info.index for quad_info in ori_quads_info_ref])
                # print("                         :", [quad_info.index for quad_info in quads_info_temp], [quad_info.index for quad_info in quads_info_ref])

                quads_info_ref_indexes = [i for i in range(len(quads_info_ref))]

                # Adjust the cross-row and cross-column information of each special quadrilateral (cell).
                for quad_info_temp in quads_info_temp:
                    # Obtain cross-row and cross-column index information
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

                    # If the index is normal
                    if index1 <= index2:
                        quad_info_temp.cache.clear()  # Clear the cache
                        quad_info_temp.cache.extend(quads_info_ref[index1 : index2 + 1])  # Update the cache
                        new_cross = sum(map(lambda quad_info: getattr(quad_info.cross, adjust_target), quads_info_ref[index1 : index2 + 1]))

                        # print("=>>>", getattr(quad_info_temp.cross, adjust_target), new_cross)

                        # Update if the new spanning row and column is larger
                        if getattr(quad_info_temp.cross, adjust_target) < new_cross:
                            setattr(quad_info_temp.cross, adjust_target, new_cross)
                            update_quads_cross(quads_info, search_mode, adjust_target)

    # Adjust the cross-column information of special quads (cells).
    run("col")
    # Adjust the cross-line information of special quads (cells).
    run("row")


def update_quads_structure(quads_info: List[QuadInfo]) -> None:
    """
    Update the structure of each quad (cell) based on the diagram

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    ###* Forward search for cross-row and cross-column information for quadrilaterals (cells).
    update_quads_cross(quads_info, "forward", "all")

    ###* Reverse search quadrilateral (cell) across rows and columns
    update_quads_cross(quads_info, "backward", "all")

    ###* Adjust the cross-row and column-spanning information of special quads (cells).
    adjust_special_quads_cross(quads_info)

    ###* Forward search and update the row and column structure of quadrilaterals (cells).
    # Find the quadrilateral box in the upper left corner
    left_top_quad_info = min(quads_info, key=lambda quad_info: quad_info.quad[0].x + quad_info.quad[0].y)

    # Search the starting stack
    search_start_stack: List[Tuple[QuadInfo, int]] = [(left_top_quad_info, 0)]

    # Traverse the entire stack and update the row structure
    while len(search_start_stack) > 0:
        search_quad_info, row_range_start = search_start_stack.pop()

        # Current span
        cross_row = deep_search_row_range(search_quad_info, row_range_start, search_start_stack)

        if len(search_quad_info.forward.bottom) > 0:
            search_start_stack.append((search_quad_info.forward.bottom[0], row_range_start + cross_row))
        elif len(search_quad_info.backward.bottom) > 0:
            search_start_stack.append((search_quad_info.backward.bottom[0], row_range_start + cross_row))

    # Reset the search starting stack
    search_start_stack = [(left_top_quad_info, 0)]

    # Traverse the entire stack and update the column structure
    while len(search_start_stack) > 0:
        search_quad_info, col_range_start = search_start_stack.pop()

        # Current span
        cross_col = deep_search_col_range(search_quad_info, col_range_start, search_start_stack)

        if len(search_quad_info.forward.right) > 0:
            search_start_stack.append((search_quad_info.forward.right[0], col_range_start + cross_col))
        elif len(search_quad_info.backward.right) > 0:
            search_start_stack.append((search_quad_info.backward.right[0], col_range_start + cross_col))


def get_quads_structure(quads: List[List[Tuple[int, int]]], adjacent_threshold: float = 10) -> List[List[QuadInfo]]:
    """
    Get structural information for a quadrilateral list

    Autor: dreamy-xay
    Date: 2024-07-09
    """
    ###* Build a diagram
    quads_info = build_relation_map(quads, adjacent_threshold, True)

    ###* Split multiple tables
    quads_info_list = split_quads(quads_info)

    ###* Update the structure of each quad (cell) based on the diagram
    for quads_info_item in quads_info_list:
        update_quads_structure(quads_info_item)

    return quads_info_list
