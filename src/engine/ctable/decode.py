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
from engine.table.decode import IndexQueryer, polygons_decode, is_group, find_near_corner_index, dist_square
from engine.table.utils import _tranpose_and_gather_feat


def get_logic_coords(lc_logic, cell_span):
    """
    Get the corner logical coordinates of the cell

    Returns:
        - logic_coords: The corner logical coordinates of the cell, (4), where 4 is the four elements of (start_col, end_col, start_row, end_row).
    """
    # Get the cell row and column span
    col_span, row_span = max(1, int(torch.round(cell_span[0]))), max(1, int(torch.round(cell_span[1])))

    # Get the logical coordinates of the corner column row of the cell
    col_logic_coords = []
    row_logic_coords = []
    for logic in lc_logic:
        col, row = (-1, -1) if logic is None else (int(torch.round(logic[0])), int(torch.round(logic[1])))
        col_logic_coords.append(col if col >= 1 else -1)
        row_logic_coords.append(row if row >= 1 else -1)

    start_col, end_col, start_row, end_row = 0, 0, 0, 0

    # Calculate cell cross-column information
    if col_logic_coords[0] == col_logic_coords[3] and col_logic_coords[0] != -1:
        start_col = col_logic_coords[0]
        end_col = col_logic_coords[0] + col_span - 1
    elif col_logic_coords[1] == col_logic_coords[2] and col_logic_coords[1] != -1:
        start_col = col_logic_coords[1] - col_span
        end_col = col_logic_coords[1] - 1
    else:
        for i, col in enumerate(col_logic_coords):
            if col >= 1:
                if i == 0 or i == 3:
                    start_col = col_logic_coords[i]
                    end_col = col_logic_coords[i] + col_span - 1
                else:
                    start_col = col_logic_coords[i] - col_span
                    end_col = col_logic_coords[i] - 1
                break

    # Calculate cell cross-row information
    if row_logic_coords[0] == row_logic_coords[1] and row_logic_coords[0] != -1:
        start_row = row_logic_coords[0]
        end_row = row_logic_coords[0] + row_span - 1
    elif row_logic_coords[2] == row_logic_coords[3] and row_logic_coords[2] != -1:
        start_row = row_logic_coords[2] - row_span
        end_row = row_logic_coords[2] - 1
    else:
        for i, row in enumerate(row_logic_coords):
            if row >= 1:
                if i == 0 or i == 1:
                    start_row = row_logic_coords[i]
                    end_row = row_logic_coords[i] + row_span - 1
                else:
                    start_row = row_logic_coords[i] - row_span
                    end_row = row_logic_coords[i] - 1
                break

    return torch.Tensor([start_col, end_col, start_row, end_row])


def cells_decode(heatmap, reg, ct2cn, cn2ct, lc, span, center_k, corner_k, center_thresh, corner_thresh, corners=False):
    """
    Cell decoding function (only supported when batch is 1)

    Args:
        - heatmap: (batch, 2, height, width)
        - reg: Offset vector graph of the center point or corner point, (batch, 2, height, width)
        - ct2cn: Vector graph with corner point pointing to center point, (batch, 8, height, width)
        - cn2ct: Vector graph of center point pointing to corner point, (batch, 8, height, width)
        - lc: corner logical coordinates, (batch, 2, height, width)
        - span: cell span, (batch, 2, height, width)
        - center_k: The maximum number of center points
        - corner_k: The maximum number of corners
        - center_thresh: The center point threshold, if the center point score is less than this threshold, it will not participate in the subsequent calculation
        - corner_thresh: Corner threshold, if the corner score is less than this threshold, it will not be included in subsequent calculations
        - corners: Whether to return corner coordinates

    Returns:
        - cells: cell, (batch, center_k, 8), where center_k is the number of center points, and 8 is the xy coordinates of the four corners of the cell, i.e., top left, top right, bottom left, bottom right
        - cells_scores: Cell fractions, (batch, center_k, 1), where the scores are sorted from highest to lowest
        - cells_corner_count: The number of times the cell corners are optimized (batch, center_k, 2), where the last dimension has two times, the first is the number of optimized corners (up to 4), and the second is the number of repeats
    """

    # Get information about the center point
    center_scores, center_indexes, center_xs, center_ys, center_polygons = polygons_decode(heatmap[:, 0:1, :, :], ct2cn, reg, K=center_k)

    # Get information about corners
    corner_scores, corner_indexes, corner_xs, corner_ys, corner_polygons = polygons_decode(heatmap[:, 1:2, :, :], cn2ct, reg, K=corner_k)

    # Get Span
    cell_spans = _tranpose_and_gather_feat(span, center_indexes)

    # Get the logical coordinates
    corner_logics = _tranpose_and_gather_feat(lc, corner_indexes)

    # Get Polygon in CPU state
    if center_polygons.device.type != "cpu":
        center_polygons_cpu = center_polygons.cpu()
        corner_polygons_cpu = corner_polygons.cpu()
    else:
        center_polygons_cpu = center_polygons
        corner_polygons_cpu = corner_polygons

    # Index querier
    iq = IndexQueryer(center_polygons, corner_polygons, center_scores, corner_scores, center_thresh, corner_thresh)

    # Create the corrected cell
    corrected_cells = center_polygons.clone()

    # Create the number of corrections and the number of repeated corrections for cell corners
    cells_corner_count = torch.zeros(center_polygons.shape[:-1] + (2,), dtype=torch.int32)

    # Create the logical coordinates of the cell
    logic_coords = torch.zeros(center_polygons.shape[:-1] + (4,), dtype=torch.int32)

    # Traverse the center point
    for i in iq.center_indices:
        # Get the polygon corresponding to the current center point (in this case, it should be a quadrilateral)
        center_polygon = center_polygons[0, i, :].view(-1, 2)
        center_polygon_cpu = center_polygons_cpu[0, i, :].view(-1, 2)

        # Get the cell corresponding to the current center point
        corrected_cell = corrected_cells[0, i, :].view(-1, 2)

        # Record the number of cell corner corrections and the number of repeated corrections
        corner_count = 0
        repeat_corner_count = 0

        lc_logic = [None, None, None, None]

        # Traverse corners
        # for j in iq.corner_indices:
        for j in iq.query(i):
            # Get the polygon corresponding to the current corner (in this case, it should be a quadrilateral)
            corner_polygon_cpu = corner_polygons_cpu[0, j, :].view(-1, 2)

            # Determine whether the current corner belongs to the polygon corresponding to the current center point
            if is_group(center_polygon_cpu, corner_polygon_cpu):
                # Get the coordinates of the current corner
                corner_x = corner_xs[0, j, 0]
                corner_y = corner_ys[0, j, 0]

                # Gets the index of the current corner in the polygon
                index = find_near_corner_index(center_polygon, corner_x, corner_y)

                # Get the specified corner of the corrected cell
                corrected_cell_corner = corrected_cell[index]

                # Get the coordinates of the specified corner of the original cell
                origin_corner_x = center_polygon[index][0]
                origin_corner_y = center_polygon[index][1]

                # Get the coordinates of the specified corner point of the corrected cell
                corrected_corner_x = corrected_cell_corner[0]
                corrected_corner_y = corrected_cell_corner[1]

                # If the specified corner of the corrected cell and the original cell are the same, it will be corrected directly, otherwise the distance will be calculated and the nearest corner point will be corrected
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

        # Update the logical coordinates of the current cell
        logic_coords[0, i, :] = get_logic_coords(lc_logic, cell_spans[0, i])

    if corners:
        return corrected_cells, center_scores, cells_corner_count, logic_coords.to(reg.device), torch.cat([corner_xs, corner_ys, corner_scores], dim=2)

    return corrected_cells, center_scores, cells_corner_count, logic_coords.to(reg.device)
