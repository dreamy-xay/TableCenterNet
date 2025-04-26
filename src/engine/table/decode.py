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
        # Get the number of center_polygon and corner_polygon
        num_center_polygons = center_polygons.shape[1]
        num_corner_polygons = corner_polygons.shape[1]

        # Corner polygons construct interval caching and provide queries
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
    Determines whether multiple points are inside a polygon

    Args:
        - center_polygon: A NumPy array of shapes (4, 2) representing the coordinates of the polygon vertices
        - corner_polygon: A NumPy array of shape (4, 2) representing the coordinates of the point to be determined

    Returns:
        - is_group: Does a point in corner_polygon exist in center_polygon
    """
    # num_points = center_polygon.shape[0] # Number of polygon vertices (4)

    # for point in corner_polygon:
    #     crossings = 0
    #     x, y = point

    # j = num_points - 1 # j is the previous vertex
    #     for i in range(num_points):
    #         ix = center_polygon[i, 0]
    #         iy = center_polygon[i, 1]
    #         jx = center_polygon[j, 0]
    #         jy = center_polygon[j, 1]

    # # The judgment point is between two x's and the ray is made in the vertical y-axis of the point
    # if ((ix > x) != (jx > x)) and (x > (jx - ix) * (y - iy) / (jy - iy)   ix):
    #    crossings  = 1

    # # Update j
    #         j = i

    # # Odd intersections are represented internally, and even intersections are external
    #     if crossings & 1:
    #         return True

    # return False
    # Expand polygon vertices and judgment points into matrices
    center_x = center_polygon[:, 0]
    center_y = center_polygon[:, 1]

    # Use torch.roll to offset the polygon vertices by one position sequentially, simulating the "previous" position of the vertices
    # For example, center_polygon[i] and center_polygon[j] are used to calculate intersections
    prev_x = torch.roll(center_x, 1)
    prev_y = torch.roll(center_y, 1)

    # Calculate dx, dy for each edge
    dx = center_x - prev_x
    dy = center_y - prev_y + 1e-6

    # Calculate whether each point intersects each edge
    # Expand corner_polygon into (num_points, 1) for batch calculations
    x = corner_polygon[:, 0]
    y = corner_polygon[:, 1]

    # Determine if each point intersects each edge of the polygon
    t1 = (center_x > x[:, None]) != (prev_x > x[:, None])  # Determine whether the x-coordinate crosses the boundary
    t2 = x[:, None] > (dx * (y[:, None] - center_y) / dy + center_x)  # Determine the intersection point of the ray and the edge

    # Count the number of intersections
    crossings = torch.sum(t1 & t2, axis=1)

    # If the number of intersections is an odd number, it means that the point is inside the polygon
    return torch.any(crossings % 2 != 0)


def is_group_faster(center_polygon, corner_polygon):
    """To determine whether a corner of a corner polygon is inside the center point polygon, center_polygon and corner_polygon are required to enter as a tensor of (N, 2)."""
    # Traverse whether all the corners of the corner polygon are inside the center point polygon
    for i in range(corner_polygon.size(0)):
        pt = Point(corner_polygon[i])
        if pt.within(center_polygon):
            return True

    return False


def is_group(center_polygon, corner_polygon, scale_factor=1.0):
    """To determine whether a corner of a corner polygon is inside the center point polygon, center_polygon and corner_polygon are required to enter as a tensor of (N, 2)."""
    # Get the maximum coordinate value of the bounding box of the center point polygon
    ctp_xmin, ctp_xmax, ctp_ymin, ctp_ymax = center_polygon[:, 0].min(), center_polygon[:, 0].max(), center_polygon[:, 1].min(), center_polygon[:, 1].max()
    # Get the maximum coordinate value of the bounding box of the corner polygon
    cnp_xmin, cnp_xmax, cnp_ymin, cnp_ymax = corner_polygon[:, 0].min(), corner_polygon[:, 0].max(), corner_polygon[:, 1].min(), corner_polygon[:, 1].max()

    # If a corner polygon does not exist, and a corner point is inside the polygon, False is returned
    if ctp_xmin > cnp_xmax or cnp_xmin > ctp_xmax or ctp_ymin > cnp_ymax or cnp_ymin > ctp_ymax:
        return False

    # Create a center point polygon
    _center_polygon = Polygon(center_polygon)
    if scale_factor < 1.0:
        # Calculate the center point of the polygon
        centroid = _center_polygon.centroid
        # Shrink the polygon based on the center point to scale
        _center_polygon = scale(_center_polygon, xfact=scale_factor, yfact=scale_factor, origin=centroid)

    # Traverse whether all the corners of the corner polygon are inside the center point polygon
    for i in range(corner_polygon.size(0)):
        pt = Point(corner_polygon[i])
        if pt.within(_center_polygon):
            return True

    return False


def dist_square(x1, y1, x2, y2):
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def find_near_corner_index(polygon, pt_x, pt_y):
    """Querying the near corner index requires polygon input as a tensor of (N, 2)."""

    # Get the polygon coordinates corresponding to the current center point
    xs = polygon[:, 0]
    ys = polygon[:, 1]

    # Calculate the square distance
    distances_square = (xs - pt_x) ** 2 + (ys - pt_y) ** 2

    # Returns the index of the nearest corner
    return torch.argmin(distances_square)


def polygons_decode(heatmap, vec, reg, K=400):
    batch = heatmap.size(0)

    # Get the peak area of the heat map (low around and high in the middle)
    heatmap = _nms(heatmap)[0]

    # Get the top K points (center or corner) of the heatmap score ranking
    scores, indexes, _, ys, xs = _topk(heatmap, K=K)

    # Fraction format conversion
    scores = scores.view(batch, K, 1)

    # Gets the offset of the center point or corner point coordinates
    point_offset = _tranpose_and_gather_feat(reg, indexes)

    # Get the coordinates of the center point or corner point
    xs = xs.view(batch, K, 1) + point_offset[:, :, 0:1]
    ys = ys.view(batch, K, 1) + point_offset[:, :, 1:2]

    # Get a vector of the regression box that points to a center point or a corner point
    polygons_vec = _tranpose_and_gather_feat(vec, indexes)

    # Get the regression box of the center point or corner point (when xy is the center point: the four points are the coordinates of the corner points of the upper left, upper right, lower left and lower right of the cell; When xy is the corner: the four points are the coordinates of the return center point of the upper left, upper right, lower left, and lower right cells
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
    Cell decoding function (only supported when batch is 1)

    Args:
        - heatmap: (batch, 2, height, width)
        - reg: Offset vector graph of the center point or corner point, (batch, 2, height, width)
        - ct2cn: Vector graph with corner point pointing to center point, (batch, 8, height, width)
        - cn2ct: Vector graph of center point pointing to corner point, (batch, 8, height, width)
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
            # start_col = torch.round((batch_col_lc[y1, x1]   batch_col_lc[y4, x4]) / 2.0)
            # end_col = torch.round((batch_col_lc[y2, x2]   batch_col_lc[y3, x3]) / 2.0) - 1, start_col
            # start_row = torch.round((batch_row_lc[y1, x1]   batch_row_lc[y2, x2]) / 2.0)
            # end_row = torch.round((batch_row_lc[y3, x3]   batch_row_lc[y4, x4]) / 2.0) - 1, start_row

            end_col = torch.max(start_col, end_col)
            end_row = torch.max(start_row, end_row)

            batch_result.append([start_col, end_col, start_row, end_row])

        results.append(batch_result)

    return torch.Tensor(results).to(cells.device)
