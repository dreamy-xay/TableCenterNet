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
    """Split a polygon into multiple triangles, support concave polygons"""
    try:
        # Create a concave polygon
        cur_polygon = Polygon([(x, y) for x, y, _ in polygon])
    except:
        return []

    # If the polygon is invalid, try to fix it
    if not cur_polygon.is_valid:
        cur_polygon = make_valid(cur_polygon)

        if not cur_polygon.is_valid:
            return []
    try:
        # Triangulate with Shapely's triangulate
        triangles = triangulate(cur_polygon)
    except:
        return []

    point_map = {}
    for x, y, v in polygon:
        point_map[(x, y)] = v

    # Convert triangles to polygons with v-values
    new_triangles = []
    for triangle in triangles:
        try:
            if not cur_polygon.contains(triangle):
                continue
        except:
            continue
        # Get all the vertex coordinates of the triangle
        triangle_points = list(triangle.exterior.coords)
        # Add a v-value for each vertex
        try:
            triangle_v = [(x, y, point_map[(x, y)]) for (x, y) in triangle_points[:-1]]
        except:
            continue
        new_triangles.append(triangle_v)

    return new_triangles


def polygon_area(points):
    """Calculate the area of the polygon"""
    x, y = zip(*points)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def interpolate_polygons(polygons, img_size, processes=["sort"]):
    """
    Interpolation polygons

    Parameters:
        polygons: A list of polygons, each polygon is a list containing (x, y, v), where v represents the value to be interpolated
        img_size: (height, width)
        process: Preprocessing type, the optional values are "sort" and "split", "split" supports the decomposition of concave polygons into multiple triangles, and "sort" sorts by area
    Returns:
        final_image: The interpolated image
        mask: A mask that distinguishes between the interpolation and background regions
    """
    # Initialize an all-0 image and an all-false mask
    final_image = np.zeros(img_size, dtype=np.float32)
    mask = np.zeros(img_size, dtype=bool)

    processed_polygons = polygons
    # Preprocess polygons
    for process in processes:
        if process == "sort":
            # Calculate the area of each polygon
            areas = [polygon_area([(x, y) for x, y, _ in polygon]) for polygon in processed_polygons]

            # Sort by area
            processed_polygons = [polygon for _, polygon in sorted(zip(areas, processed_polygons))]
        elif process == "split":
            # Split the polygon into triangles
            new_processed_polygons = []
            for polygon in processed_polygons:
                new_processed_polygons.extend(split_polygon(polygon))
            processed_polygons = new_processed_polygons

    # Interpolate each polygon
    for polygon in processed_polygons:
        # Get bbox
        x_min = math.floor(min(x for x, y, _ in polygon))
        y_min = math.floor(min(y for x, y, _ in polygon))
        x_max = math.ceil(max(x for x, y, _ in polygon))
        y_max = math.ceil(max(y for x, y, _ in polygon))
        bbox = (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

        # Extract points and values
        points = np.array([(x - bbox[0], y - bbox[1]) for x, y, _ in polygon], dtype=np.float32)
        values = np.array([v for _, _, v in polygon], dtype=np.float32)

        # Create a mesh
        x, y = np.meshgrid(np.arange(bbox[2]), np.arange(bbox[3]))
        grid_points = np.vstack((x.ravel(), y.ravel())).T

        # Interpolation
        try:
            interpolated_values = griddata(points, values, grid_points, method="linear", fill_value=-1)
        except:
            continue

        # Overlay the interpolation result onto the final image and update the mask
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
        # Use cv2.fitLine to fit the line
        line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

        # Parse the output
        vx, vy, x0, y0 = line.flatten()

        # Calculate the parameters of the linear equation ax by c = 0
        a = float(-vy if abs(vy) > eps else 0)
        b = float(vx if abs(vx) > eps else 0)
        c = float(-(a * x0 + b * y0))
        return a, b, c


def split_unicom_area(cells, adjacent_thresh=1):
    # And check the set split
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
        # Get the Bounding Box of a Polygon
        minx, miny, maxx, maxy = polygon.bounds

        # Calculate width and height
        width = maxx - minx
        height = maxy - miny

        # Calculate the scale factor
        xfact = (width + 2 * value) / width
        yfact = (height + 2 * value) / height

        return {"geom": polygon, "xfact": xfact, "yfact": yfact, "origin": polygon.centroid}

    polygons = [Polygon([point[:2] for point in cell]) for cell in cells]
    scale_polygons = [scale(**get_polygon_scale_kargs(polygon, adjacent_thresh)) for polygon in polygons]

    for i in range(num_cells):
        for j in range(i + 1, num_cells):
            if scale_polygons[i].intersects(polygons[j]):
                union(i, j)

    # Convert the results of the merge search to a connected region
    connected_components = {}
    for i in range(num_cells):
        root = find(i)
        if root not in connected_components:
            connected_components[root] = []
        connected_components[root].append(i)

    return list(connected_components.values())


def simple_interpolate_cells(cells, img_size):
    """
    Interpolate cells in an image

    Parameters:
        cells: a list of vertices in a cell, each vertex is a tuple (x, y, col, row)
        img_size: The size of the image, which is used to initialize the image and mask
    Returns:
        final_image: The interpolated image
        mask: Mask, which is used to mark the interpolated pixels
    """
    cols = []
    rows = []

    # Get Cell Box
    for cell in cells:
        cols.append([point[:3] for point in cell])
        rows.append([[point[0], point[1], point[3]] for point in cell])

    # Generate interpolation plots
    col_final_image, col_mask = interpolate_polygons(cols, img_size, ["sort"])
    row_final_image, row_mask = interpolate_polygons(rows, img_size, ["sort"])

    return np.stack([col_final_image, row_final_image], axis=0), np.stack([col_mask, row_mask], axis=0)


def interpolate_cells(cells, img_size, adjacent_thresh=1):
    """
    Interpolate cells in an image

    Parameters:
        cells: a list of vertices in a cell, each vertex is a tuple (x, y, col, row)
        img_size: The size of the image, which is used to initialize the image and mask
        adjacent_thresh: Proximity threshold, which is used to determine the boundaries of cells
    Returns:
        final_image: The interpolated image
        mask: Mask, which is used to mark the interpolated pixels
    """
    cols = []
    rows = []
    # Get a new row and column box
    for area_cells_index in split_unicom_area(cells, adjacent_thresh):
        area_cells = [cells[i] for i in area_cells_index]
        # Calculate the maximum number of columns and rows
        max_cols = max(max(point[2] for point in cell) for cell in area_cells)
        max_rows = max(max(point[3] for point in cell) for cell in area_cells)

        col_points_list = [[] for _ in range(max_cols)]
        row_points_list = [[] for _ in range(max_rows)]

        for cell in area_cells:
            for point in cell:
                col_points_list[point[2] - 1].append(point)
                row_points_list[point[3] - 1].append(point)

        # Sorting
        for points in col_points_list:
            points.sort(key=lambda pt: pt[1])
        for points in row_points_list:
            points.sort(key=lambda pt: pt[0])

        # Get Boundary Lines
        upper_line, upper_bound = Line(row_points_list[0]), row_points_list[0]
        lower_line, lower_bound = Line(row_points_list[-1]), row_points_list[-1]
        left_line, left_bound = Line(col_points_list[0]), col_points_list[0]
        right_line, right_bound = Line(col_points_list[-1]), col_points_list[-1]

        # Complete column coordinates
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

        # Complete the row coordinates
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

        # Reorder
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

    # Generate interpolation plots
    col_final_image, col_mask = interpolate_polygons(cols, img_size, ["split"])
    row_final_image, row_mask = interpolate_polygons(rows, img_size, ["split"])

    return np.stack([col_final_image, row_final_image], axis=0), np.stack([col_mask, row_mask], axis=0)


def multitable_interpolate_cells(cells, img_size):
    """
    Interpolate cells in an image

    Parameters:
        cells: a list of vertices of a cell [corner1, ..., table_id], each vertex is a tuple (x, y, col, row)
        img_size: The size of the image, which is used to initialize the image and mask
    Returns:
        final_image: The interpolated image
        mask: Mask, which is used to mark the interpolated pixels
    """
    cols = []
    rows = []

    # Split the table area
    tables = {}
    for cell in cells:
        table_id = cell[-1]
        if table_id not in tables:
            tables[table_id] = []

        tables[table_id].append(cell[:-1])

    # Get a new row and column box
    for area_cells in tables.values():
        # Calculate the maximum number of columns and rows
        max_cols = max(max(point[2] for point in cell) for cell in area_cells)
        max_rows = max(max(point[3] for point in cell) for cell in area_cells)

        col_points_list = [[] for _ in range(max_cols)]
        row_points_list = [[] for _ in range(max_rows)]

        for cell in area_cells:
            for point in cell:
                col_points_list[point[2] - 1].append(point)
                row_points_list[point[3] - 1].append(point)

        # Sorting
        for points in col_points_list:
            points.sort(key=lambda pt: pt[1])
        for points in row_points_list:
            points.sort(key=lambda pt: pt[0])

        # Get Boundary Lines
        upper_line, upper_bound = Line(row_points_list[0]), row_points_list[0]
        lower_line, lower_bound = Line(row_points_list[-1]), row_points_list[-1]
        left_line, left_bound = Line(col_points_list[0]), col_points_list[0]
        right_line, right_bound = Line(col_points_list[-1]), col_points_list[-1]

        # Complete column coordinates
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

        # Complete the row coordinates
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

        # Reorder
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

    # Generate interpolation plots
    col_final_image, col_mask = interpolate_polygons(cols, img_size, ["split"])
    row_final_image, row_mask = interpolate_polygons(rows, img_size, ["split"])

    return np.stack([col_final_image, row_final_image], axis=0), np.stack([col_mask, row_mask], axis=0)


def line_interpolation(img, p1, p2, value):
    """Interpolate pixels between two points in an image"""
    # Get the coordinates of two points
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]

    # Calculate the step size
    num_step = max(abs(x2 - x1), abs(y2 - y1))
    step_x = (x2 - x1) / (num_step + 1e-6)
    step_y = (y2 - y1) / (num_step + 1e-6)

    # Get image width and height
    img_height, img_width = img.shape[:2]

    # All intermediate points are interpolated based on distance
    for i in range(num_step + 1):
        x = round(x1 + i * step_x)
        y = round(y1 + i * step_y)

        # Make sure the coordinates are within the image range
        if 0 <= x < img_width and 0 <= y < img_height:
            img[y, x] = value  # Interpolation

    return img


def interpolate_cells_edges(cells, img_size, extend_pixel=8):
    """
    Interpolate the edges of cells in an image

    Parameters:
        cells: a list of vertices in a cell, each vertex is a tuple (x, y, col, row)
        img_size: The size of the image, which is used to initialize the image and mask
        extend_pixel: Expands the number of pixels, which is used to expand the edges of the cell
    Returns:
        final_image: The interpolated image
        mask: Mask, which is used to mark the interpolated pixels
    """
    # Initialize an all-0 image and an all-false mask
    col_final_image = np.zeros(img_size, dtype=np.float32)
    row_final_image = np.zeros(img_size, dtype=np.float32)
    mask = np.zeros(img_size, dtype=np.uint8)

    for area_cells_index in split_unicom_area(cells):
        area_cells = [cells[i] for i in area_cells_index]
        # Calculate the maximum number of columns and rows
        max_cols = max(max(point[2] for point in cell) for cell in area_cells)
        max_rows = max(max(point[3] for point in cell) for cell in area_cells)

        col_points_list = [[] for _ in range(max_cols)]
        row_points_list = [[] for _ in range(max_rows)]

        for cell in area_cells:
            for point in cell:
                col_points_list[point[2] - 1].append(point)
                row_points_list[point[3] - 1].append(point)

        # Sorting
        for points in col_points_list:
            points.sort(key=lambda pt: pt[1])
        for points in row_points_list:
            points.sort(key=lambda pt: pt[0])

        # Get Boundary Lines
        upper_line, upper_bound = Line(row_points_list[0]), row_points_list[0]
        lower_line, lower_bound = Line(row_points_list[-1]), row_points_list[-1]
        left_line, left_bound = Line(col_points_list[0]), col_points_list[0]
        right_line, right_bound = Line(col_points_list[-1]), col_points_list[-1]

        # Complete column coordinates
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

        # Complete the row coordinates
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

        # Reorder
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

        # Update the mask
        cv2.fillPoly(mask, area, 1.0)
        cv2.polylines(mask, area, True, 1.0, thickness=extend_pixel * 2 + 1)

        # Column interpolation
        for col, points in enumerate(col_points_list):
            col += 1
            for j in range(len(points) - 1):
                line_interpolation(col_final_image, points[j], points[j + 1], col)

        # Row interpolation
        for row, points in enumerate(row_points_list):
            row += 1
            for j in range(len(points) - 1):
                line_interpolation(row_final_image, points[j], points[j + 1], row)

    return np.stack([col_final_image, row_final_image], axis=0), np.stack([mask, mask], axis=0)
