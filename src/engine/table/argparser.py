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
        parser.add_argument("--center_k", type=int, default=3000, help="The maximum number of center points.")
        parser.add_argument("--corner_k", type=int, default=5000, help="The maximum number of corners.")
        parser.add_argument("--center_thresh", type=float, default=0.2, help="TopK threshold at the center point.")
        parser.add_argument("--corner_thresh", type=float, default=0.3, help="Corner TopK threshold.")
        parser.add_argument("--nms", action="store_true", help="Whether to enable NMS.")
        parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for NMS.")
        parser.add_argument("--save_corners", action="store_true", help="Whether to save the corner image.")
        parser.add_argument("--padding", type=int, default=0, help="Enter whether the image needs to be bounded in all four directions.")

    def add_val_args(self, parser):
        parser.add_argument("--resolution", type=int, default=0, help="The resolution of the input image during inference. When it is 0, it means using the default resolution of the prediction code.")
        parser.add_argument("--center_k", type=int, default=3000, help="The maximum number of center points.")
        parser.add_argument("--corner_k", type=int, default=5000, help="The maximum number of corners.")
        parser.add_argument("--center_thresh", type=float, default=0.2, help="TopK threshold at the center point.")
        parser.add_argument("--corner_thresh", type=float, default=0.3, help="Corner TopK threshold.")
        parser.add_argument("--nms", action="store_true", help="Whether to enable NMS.")
        parser.add_argument("--iou_thresh", type=float, default=0.5, help="IoU threshold for NMS.")
        parser.add_argument("--evaluate_ious", type=str, default="0.5,0.6,0.7,0.8,0.9,0.95", help="IoU that needs to be assessed at the time of assessment.")
        parser.add_argument("--evaluate_poly_iou", action="store_true", help="Whether to evaluate the polygon IoU or not the rectangular IoU.")
        parser.add_argument("--padding", type=int, default=0, help="Enter whether the image needs to be bounded in all four directions.")

    def parse_train_args(self, args):
        # Set the number of steps to attenuate the learning rate, and specify the number of steps to reduce the learning rate by 10 times
        args.lr_step = [int(i) for i in args.lr_step.split(",")]

    def parse_predict_args(self, args):
        if args.save_corners:
            # Set the path to save the corner plot in the experiment
            self.args.save_corners_dir = os.path.join(self.args.save_shows_dir, "corners")
            # Create a directory
            os.makedirs(self.args.save_corners_dir, exist_ok=True)

    def parse_val_args(self, args):
        # Parse and evaluate the IOU threshold
        args.evaluate_ious = [float(i) for i in args.evaluate_ious.split(",")]
