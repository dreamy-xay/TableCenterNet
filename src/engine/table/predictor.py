#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 10:34:01
LastEditors: dreamy-xay
LastEditTime: 2024-10-29 13:55:01
"""
import torch
import cv2
import numpy as np
import os
import json
from engine.base.predictor import BasePredictor
from utils.model import create_model, load_model
from utils.image import get_affine_transform, transform_preds, padding_image
from utils.logger import Logger
from .decode import cells_decode, logic_coords_decode
from .utils import _pnms
from .debugger import Debugger

try:
    from thop import profile

    HAS_THOP = True
except:
    HAS_THOP = False


class TablePredictor(BasePredictor):
    resolution = [1024, 1024]  # 数据集最终输入图片分辨率
    mean = [0.40789654, 0.44719302, 0.47026115]  # 标准化均值
    std = [0.28863828, 0.27408164, 0.27809835]  # 标准化标准差

    down_ratio = 4  # 下采样倍数

    def __init__(self, args):
        self.args = args

        # 更新推理图像分辨率
        resolution = getattr(args, "resolution", 0)
        if resolution != 0:
            self.resolution = [resolution, resolution]

        # 加载训练好的模型
        model = create_model(args.model)
        model = load_model(model, args.model_path)  # 加载 model_best
        if isinstance(model, tuple):  # 如果是加载的 model_last 等，则只加载 model
            model = model[0]

        # 初始化构造函数
        super().__init__(model)

        # 标准化数据预处理
        self.mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)

        # 单元格分数衰减阈值
        self.cell_min_optimize_count = 2
        self.cell_decay_thresh = 0.4

        # 调试器
        self.debugger = Debugger()

        # 保存参数
        Logger.save_options(args)

        # 打印 GFLOPs和参数量
        if HAS_THOP:
            with torch.no_grad():
                flops, params = profile(self.model, inputs=(torch.randn(1, 3, *self.resolution),))
                print(f"Model GFLOPs: {flops / 1e9}, Model Parameters: {params}")

    def pre_process(self, image, image_name, *args, **kwargs):
        input_height, input_width = self.resolution
        if self.args.padding > 0:
            image = padding_image(image, self.args.padding)
            shift = np.array([self.args.padding, self.args.padding], dtype=np.float32)
        else:
            shift = np.array([0, 0], dtype=np.float32)

        height, width = image.shape[:2]
        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        scale = max(height, width) * 1.0

        # 获取仿射变换矩阵
        trans_input = get_affine_transform(center, scale, 0, [input_width, input_height])
        # 获取输入图像
        input_image = cv2.warpAffine(image, trans_input, (input_width, input_height), flags=cv2.INTER_LINEAR)

        # 标准化数据
        input_image = ((input_image.astype(np.float32) / 255.0 - self.mean) / self.std).astype(np.float32)

        # 一张图片做成一个 batch
        batch_input = input_image.transpose(2, 0, 1).reshape(1, 3, input_height, input_width)

        batch_input = torch.from_numpy(batch_input)

        # 预测结果还原信息
        meta = {
            "center": center,
            "scale": scale,
            "rotate": 0,
            "shift": shift,
            "input_height": input_height,
            "input_width": input_width,
            "output_height": input_height // self.down_ratio,
            "output_width": input_width // self.down_ratio,
            "image_name": image_name,
        }

        return batch_input, meta

    def process(self, input, meta, *args, **kwargs):
        with torch.no_grad():
            # 模型推理
            outputs = self.model(input)
            output = outputs[-1]

            # 获取模型推理输出层
            hm = output["hm"].sigmoid_()
            reg = output["reg"]
            ct2cn = output["ct2cn"]
            cn2ct = output["cn2ct"]
            lc = output["lc"]

            # 输出推理结果图
            np.save(os.path.join(self.args.save_dir, meta["image_name"]), lc.detach().cpu()[0].numpy())

            # 单元格物理坐标解码
            cells, cells_scores, cells_corner_count, *rets = cells_decode(
                hm, reg, ct2cn, cn2ct, self.args.center_k, self.args.corner_k, self.args.center_thresh, self.args.corner_thresh, self.args.save_corners
            )

            # 单元格逻辑坐标解码
            logic_coords = logic_coords_decode(lc, cells)

            # 根据单元格的角点优化次数降低其评分
            is_modify = False
            for i in range(cells.size(1)):
                if cells_scores[0, i, 0] < self.args.center_thresh:
                    break

                if cells_corner_count[0, i, :].sum() <= self.cell_min_optimize_count:
                    cells_scores[0, i, 0] *= self.cell_decay_thresh
                    is_modify = True

            # 合并输出
            detections = torch.cat([cells, cells_scores, logic_coords], dim=2)

            # 如果修改了score则重新排序
            if is_modify:
                _, sorted_inds = torch.sort(cells_scores, descending=True, dim=1)
                detections = detections.gather(1, sorted_inds.expand_as(detections))

            # 返回检测结果
            return detections, rets[0] if self.args.save_corners else None, meta

    def post_process(self, detections, corners, meta, *args, **kwargs):
        # 获取模型预测结果
        detections = detections.detach().cpu().numpy()[0]

        # 删除置信度低于阈值的单元格
        detections = detections[detections[:, 8] >= self.args.center_thresh]

        # 获取还原预测结果的元数据
        center, scale, rotate, shift, output_width, output_height = meta["center"], meta["scale"], meta["rotate"], meta["shift"], meta["output_width"], meta["output_height"]

        # 还原预测结果
        detections[:, 0:2] = transform_preds(detections[:, 0:2], center, scale, (output_width, output_height), rotate, shift)
        detections[:, 2:4] = transform_preds(detections[:, 2:4], center, scale, (output_width, output_height), rotate, shift)
        detections[:, 4:6] = transform_preds(detections[:, 4:6], center, scale, (output_width, output_height), rotate, shift)
        detections[:, 6:8] = transform_preds(detections[:, 6:8], center, scale, (output_width, output_height), rotate, shift)

        # 非极大抑制
        if self.args.nms:
            detections = _pnms(detections, self.args.iou_thresh, True)

        # 还原角点
        if self.args.save_corners:
            corners = corners.detach().cpu().numpy()[0]
            corners = corners[corners[:, 2] >= self.args.corner_thresh]
            corners[:, 0:2] = transform_preds(corners[:, 0:2], center, scale, (output_width, output_height), rotate)

            return detections, corners

        return detections, None

    def generate(self, image, result):
        # 如果需要保存或者显示，则生成可视化结果
        if self.args.save or self.args.show:
            detections = result[0]

            result_image = self.debugger.process_image(image)
            scale, pad = self.debugger.IMAGE_SCALE, self.debugger.IMAGE_PAD

            self.debugger.draw_polygons(result_image, detections[:, :8] * scale + pad, color=(0, 0, 255))
            self.debugger.draw_logic_coords(result_image, detections[:, 9:13], detections[:, 0:2] * scale + pad)

            return result_image

        return None

    def run(self):
        devices = self._get_devices(self.args.device)  # 获取推理所需GPU
        is_parallel_infer = self.args.workers * len(devices) > 1  # 是否并行推理

        if is_parallel_infer and os.path.isdir(self.args.source):
            print(f"Start multi-process prediction. Using GPUs {devices}, and each GPU runs {self.args.workers} processes in parallel.")
            setattr(self.args, "devices", devices)  # 设置多进程设备列表
            results = self.parallel_predict(self.args)
        else:
            if is_parallel_infer:
                print("The input source is a file rather than a directory. Switch to single-process prediction.")
            else:
                print(f"Start single-process prediction. Use GPU {self.args.device}.")

            results = self.predict(self.args)

        # 保存预测结果
        if self.args.save_result:
            self.save_results(results, self.args.save_results_dir)

        # 保存角点图片
        if self.args.save_corners:
            self.save_corners(results, self.args.save_corners_dir)

    def save_results(self, results, save_path):
        def get_cells(polygons):
            cells = []
            for polygon in polygons:
                polygon = [float(num) for num in polygon]
                cell = {"x1": polygon[0], "y1": polygon[1], "x2": polygon[2], "y2": polygon[3], "x3": polygon[4], "y3": polygon[5], "x4": polygon[6], "y4": polygon[7]}
                cells.append({"cell": cell, "score": polygon[8], "start_col": int(polygon[9]), "end_col": int(polygon[10]), "start_row": int(polygon[11]), "end_row": int(polygon[12])})
            return cells

        for result in results:
            # 保存json文件
            type = result["type"]
            name = result["name"]
            detections = result["result"]

            json_path = os.path.join(save_path, os.path.splitext(name)[0] + ".json")

            if type == "video":
                content = {"type": type, "file_name": name, "frames": []}
                for index, frame in enumerate(detections):
                    content["frames"].append({"frame_id": index, "cells": get_cells(frame[0])})
            elif type == "image":
                content = {"type": type, "file_name": name, "cells": get_cells(detections[0])}
            else:
                raise ValueError("Unknown type: {}".format(type))

            with open(json_path, "w") as f:
                json.dump(content, f)

    def save_corners(self, results, save_path):
        for result in results:
            corners = result["result"][1]
            if corners is not None and "image" in result:
                image = result["image"]
                name = result["name"]
                corners_image_path = os.path.join(save_path, os.path.splitext(name)[0] + ".jpg")
                corners_image = image.copy()
                self.debugger.draw_points(corners_image, corners, (0, 0, 255), 4)
                cv2.imwrite(corners_image_path, corners_image)
