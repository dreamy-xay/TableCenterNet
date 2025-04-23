#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 10:35:11
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 10:42:26
"""
import os
from engine.base.argparser import BaseArgParser


class TableArgParser(BaseArgParser):
    def __init__(self):
        super().__init__()

    def add_train_args(self, parser):
        parser.add_argument("--seed", type=int, default=317, help="Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.")
        parser.add_argument("--lr", type=float, default=1.25e-4, help="The learning rate required for model training.")
        parser.add_argument(
            "--lr_step",
            type=str,
            default="",
            help="The number of training rounds for which the learning rate is decayed, each time the learning rate is decayed by 10 times based on the current learning rate.",
        )
        parser.add_argument("--val_epochs", type=int, default=10, help="The number of training epochs required for verification.")

    def add_predict_args(self, parser):
        parser.add_argument("--resolution", type=int, default=0, help="The resolution of the input image during inference. When it is 0, it means using the default resolution of the prediction code.")
        parser.add_argument("--center_k", type=int, default=3000, help="")
        parser.add_argument("--corner_k", type=int, default=5000, help="")
        parser.add_argument("--center_thresh", type=float, default=0.2, help="")
        parser.add_argument("--corner_thresh", type=float, default=0.3, help="")
        parser.add_argument("--nms", action="store_true", help="")
        parser.add_argument("--iou_thresh", type=float, default=0.5, help="")
        parser.add_argument("--save_corners", action="store_true", help="")
        parser.add_argument("--padding", type=int, default=0, help="")

    def add_val_args(self, parser):
        parser.add_argument("--resolution", type=int, default=0, help="The resolution of the input image during inference. When it is 0, it means using the default resolution of the prediction code.")
        parser.add_argument("--center_k", type=int, default=3000, help="")
        parser.add_argument("--corner_k", type=int, default=5000, help="")
        parser.add_argument("--center_thresh", type=float, default=0.2, help="")
        parser.add_argument("--corner_thresh", type=float, default=0.3, help="")
        parser.add_argument("--nms", action="store_true", help="")
        parser.add_argument("--iou_thresh", type=float, default=0.5, help="")
        parser.add_argument("--evaluate_ious", type=str, default="0.5,0.6,0.7,0.8,0.9,0.95", help="")
        parser.add_argument("--evaluate_poly_iou", action="store_true", help="")
        parser.add_argument("--padding", type=int, default=0, help="")

    def parse_train_args(self, args):
        # 设置学习率衰减步数，指定步数学习率衰减10倍
        args.lr_step = [int(i) for i in args.lr_step.split(",")]

    def parse_predict_args(self, args):
        if args.save_corners:
            # 设置实验中角点图的保存路径
            self.args.save_corners_dir = os.path.join(self.args.save_shows_dir, "corners")
            # 创建目录
            os.makedirs(self.args.save_corners_dir, exist_ok=True)

    def parse_val_args(self, args):
        # 解析评估IOU阈值
        args.evaluate_ious = [float(i) for i in args.evaluate_ious.split(",")]
