#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
Version:
Autor: dreamy-xay
Date: 2024-10-22 20:21:30
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 20:22:08
"""
import numpy as np
import os
import cv2
import math
from data.COCO import COCO
from utils.image import affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian


class WTWDataset(COCO):
    num_classes = 2  # There are two types of targets: cell center point and cell corner point

    def __init__(self, data_yaml, split):
        super(WTWDataset, self).__init__(data_yaml, split)

    def __getitem__(self, index):
        # Get the name of the image and the corresponding tag of the image
        image_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[image_id])[0]["file_name"]
        image_path = os.path.join(self.image_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        # Get the number of objects in the image
        num_objects = min(len(anns), self.max_objects)

        # Read the image
        image = cv2.imread(image_path)

        # Image Enhancement
        input_image, _, trans_output, output_h, output_w = self._augmentation(image)

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objects * 5, 2), dtype=np.float32)  # 5 Each element represents the offset of the center point coordinates and the four corner point coordinates
        ct2cn = np.zeros(
            (self.max_objects, 8), dtype=np.float32
        )  # 8 elements represent the offset of the coordinates of the center point of the cell relative to the coordinates of the 4 corner points
        cn2ct = np.zeros((self.max_corners, 8), dtype=np.float32)  # 8 elements represent the offset of the 4 corner coordinates relative to the center point coordinates of the cell
        reg_ind = np.zeros((self.max_objects * 5), dtype=np.int64)
        reg_mask = np.zeros((self.max_objects * 5), dtype=np.uint8)
        ct_ind = np.zeros((self.max_objects), dtype=np.int64)
        ct_mask = np.zeros((self.max_objects), dtype=np.uint8)
        cn_ind = np.zeros((self.max_corners), dtype=np.int64)
        cn_mask = np.zeros((self.max_corners), dtype=np.uint8)
        ct_cn_ind = np.zeros((self.max_objects * 4), dtype=np.int64)  # Record the index of the jth corner of cell in the corner vector
        lc_coords = np.zeros((self.max_objects, 2), dtype=np.float32)
        lc_span = np.zeros((self.max_objects, 2), dtype=np.float32)

        # Corner Dictionary
        corner_dict = {}

        # Enumerate all cells
        for i in range(num_objects):
            ann = anns[i]

            # Remove the corner points from the label
            seg_mask = ann["segmentation"][0]
            x1, y1 = seg_mask[0], seg_mask[1]
            x2, y2 = seg_mask[2], seg_mask[3]
            x3, y3 = seg_mask[4], seg_mask[5]
            x4, y4 = seg_mask[6], seg_mask[7]

            # Corner coordinates
            corners = np.array([x1, y1, x2, y2, x3, y3, x4, y4])

            # Corner coordinate transformation
            corners[0:2] = affine_transform(corners[0:2], trans_output)
            corners[2:4] = affine_transform(corners[2:4], trans_output)
            corners[4:6] = affine_transform(corners[4:6], trans_output)
            corners[6:8] = affine_transform(corners[6:8], trans_output)
            corners[[0, 2, 4, 6]] = np.clip(corners[[0, 2, 4, 6]], 0, output_w - 1)
            corners[[1, 3, 5, 7]] = np.clip(corners[[1, 3, 5, 7]], 0, output_h - 1)

            # Determine whether a corner can form a valid cell
            if not self._is_effective_quad(corners):
                continue

            # Get cell bbox information
            max_x = max(corners[[0, 2, 4, 6]])
            min_x = min(corners[[0, 2, 4, 6]])
            max_y = max(corners[[1, 3, 5, 7]])
            min_y = min(corners[[1, 3, 5, 7]])
            bbox_h, bbox_w = max_y - min_y, max_x - min_x

            if bbox_h > 0 and bbox_w > 0:
                # * Cell center point processing
                # Calculate the heatmap radius parameter
                radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
                radius = max(0, int(radius))

                # Calculate the center of the cell
                cell_center = np.array([(max_x + min_x) / 2.0, (max_y + min_y) / 2.0], dtype=np.float32)
                cell_center_int = cell_center.astype(np.int32)

                # offset: record the coordinate offset of the center point of the cell; Record the index of the center point of the cell in the picture vector; Record whether the cell center point is valid or not
                reg[i] = cell_center - cell_center_int
                reg_ind[i] = cell_center_int[1] * output_w + cell_center_int[0]
                reg_mask[i] = 1

                # heatmap: record the index of the center point of the cell in the image vector; Record whether the cell center point is valid or not
                ct_ind[i] = cell_center_int[1] * output_w + cell_center_int[0]
                ct_mask[i] = 1

                # Draw a heat map of the center point of the cell
                draw_umich_gaussian(hm[0], cell_center_int, radius)

                # Calculate the offset of the center of the cell relative to the corner point
                ct2cn[i] = (
                    cell_center[0] - corners[0],
                    cell_center[1] - corners[1],
                    cell_center[0] - corners[2],
                    cell_center[1] - corners[3],
                    cell_center[0] - corners[4],
                    cell_center[1] - corners[5],
                    cell_center[0] - corners[6],
                    cell_center[1] - corners[7],
                )

                # Get the logical coordinates
                start_col, end_col, start_row, end_row = [int(coord) + 1 for coord in ann["logic_axis"][0]]

                # Get the start column information of the start row
                lc_coords[i][0], lc_coords[i][1] = start_col, start_row

                # Obtain cross-row and cross-column information
                lc_span[i][0], lc_span[i][1] = end_col - start_col + 1, end_row - start_row + 1

                # * Enumerate every corner point
                for j in range(4):
                    start_index = j * 2
                    end_index = start_index + 2
                    corner = np.array(corners[start_index:end_index], dtype=np.float32)

                    corner_int = corner.astype(np.int32)
                    corner_key = f"{corner_int[0]}_{corner_int[1]}"

                    # Make sure that each corner is added only once
                    if corner_key not in corner_dict:
                        # Add a dictionary
                        num_corner = len(corner_dict)
                        corner_dict[corner_key] = num_corner

                        # offset: record the coordinate offset of the corner point, the first max_objects is the coordinate offset of the center point of the cell; Record the index of the corner points in the image vector; Record whether the corners are valid
                        reg[self.max_objects + num_corner] = np.array([abs(corner[0] - corner_int[0]), abs(corner[1] - corner_int[1])])
                        reg_ind[self.max_objects + num_corner] = corner_int[1] * output_w + corner_int[0]
                        reg_mask[self.max_objects + num_corner] = 1
                        # heatmap: record the index of the corners in the image vector; Record whether the corners are valid
                        cn_ind[num_corner] = corner_int[1] * output_w + corner_int[0]
                        cn_mask[num_corner] = 1

                        # Draw a corner heat map
                        draw_umich_gaussian(hm[1], corner_int, 2)

                        # Record the offset of the corner point relative to the center of the cell
                        cn2ct[num_corner][start_index:end_index] = np.array([corner[0] - cell_center[0], corner[1] - cell_center[1]])

                        # Record the index of the jth corner of cell in the corner vector
                        ct_cn_ind[4 * i + j] = num_corner * 4 + j
                    else:
                        index_of_key = corner_dict[corner_key]

                        # Record the offset of the corner point relative to the center of the cell
                        cn2ct[index_of_key][start_index:end_index] = np.array([corner[0] - cell_center[0], corner[1] - cell_center[1]])

                        # Record the index of the jth corner of cell in the corner vector
                        ct_cn_ind[4 * i + j] = index_of_key * 4 + j

        # Standardization
        input_image = (input_image - self.mean) / self.std
        input_image = input_image.transpose(2, 0, 1)  # HWC->CHW

        # Construct tag data
        element = {
            "input": input_image,
            "hm": hm,
            "ct_ind": ct_ind,
            "ct_mask": ct_mask,
            "cn_ind": cn_ind,
            "cn_mask": cn_mask,
            "reg": reg,
            "reg_ind": reg_ind,
            "reg_mask": reg_mask,
            "ct2cn": ct2cn,
            "cn2ct": cn2ct,
            "ct_cn_ind": ct_cn_ind,
            "lc_coords": lc_coords,
            "lc_span": lc_span,
        }

        return element


def get_dataset(args):
    return WTWDataset
