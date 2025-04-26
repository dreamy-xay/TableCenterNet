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
import pycocotools.coco as coco
import numpy as np
import os
import cv2
import yaml
from torch.utils.data import Dataset
from cfg import ROOT_DIR, DEFAULT_CFG
from utils.image import color_aug
from utils.image import get_affine_transform


class COCO(Dataset):
    def __init__(self, data_yaml, split):
        super(COCO, self).__init__()
        # Load the configuration file
        self._load_cfg(data_yaml)

        # Parameter initialization
        self.is_train = split == "train"

        # Data loading
        self.coco = coco.COCO(os.path.join(self.label_dir, self.label_name[split]))
        self.images = self.coco.getImgIds()
        print("Loaded {} {} samples of data set {}.".format(split, len(self.images), self.__class__.__name__))

    def _load_cfg(self, cfg_yaml):
        with open(cfg_yaml, "r") as f:
            cfg = yaml.safe_load(f)

            # * Dataset path
            path_cfg = cfg["path"]
            if path_cfg["reset_data_dir"]:
                self.datasets_dir = os.path.join(path_cfg["reset_data_dir"], path_cfg["dataset_name"])
            else:
                self.datasets_dir = os.path.join(ROOT_DIR, DEFAULT_CFG["dirs"]["data_dir"], path_cfg["dataset_name"])
            self.image_dir = os.path.join(self.datasets_dir, path_cfg.get("reset_images_dir") if path_cfg.get("reset_images_dir") else "images")
            self.label_dir = os.path.join(self.datasets_dir, "labels")
            print(f"Image dir: {self.image_dir}, Label dir: {self.label_dir}")
            self.label_name = {"train": path_cfg["train"], "val": path_cfg["val"], "test": path_cfg["test"]}

            # * Model parameters
            model_cfg = cfg["model"]
            self.down_ratio = model_cfg["down_ratio"]  # Downsampling multiplier

            # * Dataset-related parameters
            dataset_cfg = cfg["dataset"]
            self.resolution = dataset_cfg["resolution"]  # Enter the image resolution

            # * Limit on the number of targets
            limit_cfg = cfg["limit"]
            self.max_objects = limit_cfg["max_objects"]  # Maximum number of cells
            self.max_corners = limit_cfg["max_corners"]  # Maximum number of corners

            # * Normalized parameters
            stand_cfg = cfg["stand"]
            self.mean = np.array(stand_cfg["mean"], dtype=np.float32).reshape(1, 1, 3)
            self.std = np.array(stand_cfg["std"], dtype=np.float32).reshape(1, 1, 3)

            # * Data augmentation parameters
            aug_cfg = cfg["augmentation"]
            # Color Enhancement
            color_aug_cfg = aug_cfg["color"]
            self.color_aug = color_aug_cfg["enable"]
            self.color_aug_params = np.random.RandomState(123), np.array(color_aug_cfg["eigval"], dtype=np.float32), np.array(color_aug_cfg["eigvec"], dtype=np.float32)
            # Affine transformation
            self.random_crop_aug = aug_cfg["random_crop"]
            self.scale = aug_cfg["scale"]
            self.shift = aug_cfg["shift"]
            self.rotate = aug_cfg["rotate"]

    def __len__(self):
        return len(self.images)

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _is_effective_quad(self, quad):
        count_x = len(set([quad[0], quad[2], quad[4], quad[6]]))
        count_y = len(set([quad[1], quad[3], quad[5], quad[7]]))

        return count_x >= 2 and count_y >= 2

    def _augmentation(self, image):
        height, width = image.shape[:2]

        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        scale = max(height, width) * 1.0  # The maximum width and height of the initial storage
        rot = np.random.randint(-self.rotate, self.rotate) if self.rotate > 0 else 0
        input_h, input_w = self.resolution

        if self.is_train:
            if self.random_crop_aug:
                w_border = self._get_border(128, width)
                h_border = self._get_border(128, height)
                center[0] = np.random.randint(low=w_border, high=width - w_border)
                center[1] = np.random.randint(low=h_border, high=height - h_border)
                scale *= np.random.choice(np.arange(1 - self.scale, 1 + self.scale, 0.1)) if self.scale > 0 else 1.0
            else:
                center[0] += scale * np.random.uniform(-self.shift, self.shift)
                center[1] += scale * np.random.uniform(-self.shift, self.shift)
                scale *= np.random.choice(np.arange(1 - self.scale, 1 + self.scale, 0.1)) if self.scale > 0 else 1.0

        output_h = input_h // self.down_ratio
        output_w = input_w // self.down_ratio

        trans_input = get_affine_transform(center, scale, rot, [input_w, input_h])
        trans_output = get_affine_transform(center, scale, rot, [output_w, output_h])

        input_image = cv2.warpAffine(image, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        input_image = input_image.astype(np.float32) / 255.0
        if self.is_train and self.color_aug:
            color_aug(input_image, *self.color_aug_params)

        return input_image, trans_input, trans_output, output_h, output_w

    def __getitem__(self, index):
        raise NotImplementedError("Method __getitem__ must be implemented in subclass.")
