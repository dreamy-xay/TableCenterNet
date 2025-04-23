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
    num_classes = 2  # 两类目标：单元格中心点、单元格角点

    def __init__(self, data_yaml, split):
        super(WTWDataset, self).__init__(data_yaml, split)

    def __getitem__(self, index):
        # 获取图像名称和图像对应标签
        image_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[image_id])[0]["file_name"]
        image_path = os.path.join(self.image_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[image_id])
        anns = self.coco.loadAnns(ids=ann_ids)

        # 获取图像中物体数量
        num_objects = min(len(anns), self.max_objects)

        # 读取图像
        image = cv2.imread(image_path)

        # 图像增强
        input_image, _, trans_output, output_h, output_w = self._augmentation(image)

        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objects * 5, 2), dtype=np.float32)  # 5各元素表示中心点坐标和四个角点坐标的offset
        ct2cn = np.zeros((self.max_objects, 8), dtype=np.float32)  # 8个元素表示单元格中心点坐标相对4个角点坐标的偏移量
        cn2ct = np.zeros((self.max_corners, 8), dtype=np.float32)  # 8个元素表示4个角点坐标相对单元格中心点坐标的偏移量
        reg_ind = np.zeros((self.max_objects * 5), dtype=np.int64)
        reg_mask = np.zeros((self.max_objects * 5), dtype=np.uint8)
        ct_ind = np.zeros((self.max_objects), dtype=np.int64)
        ct_mask = np.zeros((self.max_objects), dtype=np.uint8)
        cn_ind = np.zeros((self.max_corners), dtype=np.int64)
        cn_mask = np.zeros((self.max_corners), dtype=np.uint8)
        ct_cn_ind = np.zeros((self.max_objects * 4), dtype=np.int64)  # 记录第i个单元格的第j个角点在角点向量中的索引
        lc_coords = np.zeros((self.max_objects, 2), dtype=np.float32)
        lc_span = np.zeros((self.max_objects, 2), dtype=np.float32)

        # 角点字典
        corner_dict = {}

        # 枚举所有单元格
        for i in range(num_objects):
            ann = anns[i]

            # 从标签中取出角点
            seg_mask = ann["segmentation"][0]
            x1, y1 = seg_mask[0], seg_mask[1]
            x2, y2 = seg_mask[2], seg_mask[3]
            x3, y3 = seg_mask[4], seg_mask[5]
            x4, y4 = seg_mask[6], seg_mask[7]

            # 角点坐标
            corners = np.array([x1, y1, x2, y2, x3, y3, x4, y4])

            # 角点坐标变换
            corners[0:2] = affine_transform(corners[0:2], trans_output)
            corners[2:4] = affine_transform(corners[2:4], trans_output)
            corners[4:6] = affine_transform(corners[4:6], trans_output)
            corners[6:8] = affine_transform(corners[6:8], trans_output)
            corners[[0, 2, 4, 6]] = np.clip(corners[[0, 2, 4, 6]], 0, output_w - 1)
            corners[[1, 3, 5, 7]] = np.clip(corners[[1, 3, 5, 7]], 0, output_h - 1)

            # 判断角点是否能够构成一个有效的单元格
            if not self._is_effective_quad(corners):
                continue

            # 获取单元格 bbox 信息
            max_x = max(corners[[0, 2, 4, 6]])
            min_x = min(corners[[0, 2, 4, 6]])
            max_y = max(corners[[1, 3, 5, 7]])
            min_y = min(corners[[1, 3, 5, 7]])
            bbox_h, bbox_w = max_y - min_y, max_x - min_x

            if bbox_h > 0 and bbox_w > 0:
                # * 单元格中心点处理
                # 计算热力图半径参数
                radius = gaussian_radius((math.ceil(bbox_h), math.ceil(bbox_w)))
                radius = max(0, int(radius))

                # 计算单元格中心
                cell_center = np.array([(max_x + min_x) / 2.0, (max_y + min_y) / 2.0], dtype=np.float32)
                cell_center_int = cell_center.astype(np.int32)

                # offset：记录单元格中心点坐标偏移量；记录单元格中心点在图片向量中的索引；记录单元格中心点是否有效
                reg[i] = cell_center - cell_center_int
                reg_ind[i] = cell_center_int[1] * output_w + cell_center_int[0]
                reg_mask[i] = 1

                # heatmap：记录单元格中心点在图片向量中的索引；记录单元格中心点是否有效
                ct_ind[i] = cell_center_int[1] * output_w + cell_center_int[0]
                ct_mask[i] = 1

                # 绘制单元格中心点热力图
                draw_umich_gaussian(hm[0], cell_center_int, radius)

                # 计算单元格中心相对于角点的偏移量
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

                # 获取逻辑坐标
                start_col, end_col, start_row, end_row = [int(coord) + 1 for coord in ann["logic_axis"][0]]

                # 获取开始行开始列信息
                lc_coords[i][0], lc_coords[i][1] = start_col, start_row

                # 获取跨行跨列信息
                lc_span[i][0], lc_span[i][1] = end_col - start_col + 1, end_row - start_row + 1

                # * 枚举每一个角点
                for j in range(4):
                    start_index = j * 2
                    end_index = start_index + 2
                    corner = np.array(corners[start_index:end_index], dtype=np.float32)

                    corner_int = corner.astype(np.int32)
                    corner_key = f"{corner_int[0]}_{corner_int[1]}"

                    # 保证每个角点只加入一次
                    if corner_key not in corner_dict:
                        # 加入字典
                        num_corner = len(corner_dict)
                        corner_dict[corner_key] = num_corner

                        # offset：记录角点坐标偏移量，前 max_objects 是单元格中心点坐标偏移量；记录角点在图片向量中的索引；记录角点是否有效
                        reg[self.max_objects + num_corner] = np.array([abs(corner[0] - corner_int[0]), abs(corner[1] - corner_int[1])])
                        reg_ind[self.max_objects + num_corner] = corner_int[1] * output_w + corner_int[0]
                        reg_mask[self.max_objects + num_corner] = 1
                        # heatmap：记录角点在图片向量中的索引；记录角点是否有效
                        cn_ind[num_corner] = corner_int[1] * output_w + corner_int[0]
                        cn_mask[num_corner] = 1

                        # 绘制角点热力图
                        draw_umich_gaussian(hm[1], corner_int, 2)

                        # 记录角点相对于单元格中心的偏移量
                        cn2ct[num_corner][start_index:end_index] = np.array([corner[0] - cell_center[0], corner[1] - cell_center[1]])

                        # 记录第i个单元格的第j个角点在角点向量中的索引
                        ct_cn_ind[4 * i + j] = num_corner * 4 + j
                    else:
                        index_of_key = corner_dict[corner_key]

                        # 记录角点相对于单元格中心的偏移量
                        cn2ct[index_of_key][start_index:end_index] = np.array([corner[0] - cell_center[0], corner[1] - cell_center[1]])

                        # 记录第i个单元格的第j个角点在角点向量中的索引
                        ct_cn_ind[4 * i + j] = index_of_key * 4 + j

        # 标准化
        input_image = (input_image - self.mean) / self.std
        input_image = input_image.transpose(2, 0, 1)  # HWC->CHW

        # 构造标签数据
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
