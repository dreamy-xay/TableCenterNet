#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-28 15:17:12
LastEditors: dreamy-xay
LastEditTime: 2024-10-29 15:53:42
"""
import numpy as np
import cv2
import os


class Debugger(object):
    IMAGE_PAD = 0
    PAD_COLOR = (255, 255, 255)
    IMAGE_SCALE = 1.0

    def __init__(self, ipynb=False):
        self.ipynb = ipynb
        if not self.ipynb:
            import matplotlib.pyplot as plt

            self.plt = plt

        self.imgs = {}

    def add_img(self, img, img_id, revert_color=False):
        if revert_color:
            img = 255 - img

        self.imgs[img_id] = self.process_image(img)

    @staticmethod
    def resize_image(image, scale):
        """
        Resize the image according to the zoom factor
        :param image: input image (OpenCV format)
        :param scale: Scale factor (e.g. 0.5 to shrink to half, 2 to double magnification)
        :return: Resized image
        """
        # Get the dimensions of the original image
        height, width = image.shape[:2]

        # Calculate the new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Select the interpolation method: use INTER_CUBIC when zooming in, and INTER_AREA when zooming out
        if scale > 1:
            interpolation = cv2.INTER_CUBIC  # Use cubic interpolation when zooming in
        else:
            interpolation = cv2.INTER_AREA  # Use region interpolation when zooming out (better quality)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

        return resized_image

    @staticmethod
    def process_image(image):
        if Debugger.IMAGE_SCALE != 1.0:
            image = Debugger.resize_image(image, Debugger.IMAGE_SCALE)

        if Debugger.IMAGE_PAD <= 0:
            return image.copy() if Debugger.IMAGE_SCALE == 1.0 else image
        else:
            return cv2.copyMakeBorder(image, Debugger.IMAGE_PAD, Debugger.IMAGE_PAD, Debugger.IMAGE_PAD, Debugger.IMAGE_PAD, cv2.BORDER_CONSTANT, value=Debugger.PAD_COLOR)

    def show_img(self, img_id, pause=False):
        cv2.imshow("{}".format(img_id), self.imgs[img_id])
        if pause:
            cv2.waitKey()

    def add_blend_img(self, back, fore, img_id, trans=0.7):
        if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
            fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
        if len(fore.shape) == 2:
            fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
        self.imgs[img_id] = back * (1.0 - trans) + fore * trans
        self.imgs[img_id][self.imgs[img_id] > 255] = 255
        self.imgs[img_id][self.imgs[img_id] < 0] = 0
        self.imgs[img_id] = self.imgs[img_id].astype(np.uint8).copy()

    def gen_heatmap(self, img, down_ratio, colors):
        img = img.copy()
        c, h, w = img.shape[0], img.shape[1], img.shape[2]
        output_res = (h * down_ratio, w * down_ratio)
        img = img.transpose(1, 2, 0).reshape(h, w, c, 1).astype(np.float32)
        colors = np.array(colors, dtype=np.float32).reshape(1, 1, c, 3)
        color_map = (img * colors).max(axis=2).astype(np.uint8)
        color_map = cv2.resize(color_map, (output_res[0], output_res[1]))
        return color_map

    def add_polygon(self, img_id, polygon, color):
        self.draw_polygons(self.imgs[img_id], [polygon], color=color)

    def show_all_imgs(self, pause=False):
        if not self.ipynb:
            for i, v in self.imgs.items():
                cv2.imshow("{}".format(i), v)
            if cv2.waitKey(0 if pause else 1) == 27:
                import sys

                sys.exit(0)
        else:
            self.ax = None
            nImgs = len(self.imgs)
            fig = self.plt.figure(figsize=(nImgs * 10, 10))
            for i, (_, v) in enumerate(self.imgs.items()):
                fig.add_subplot(1, nImgs, i + 1)
                if len(v.shape) == 3:
                    self.plt.imshow(cv2.cvtColor(v, cv2.COLOR_BGR2RGB))
                else:
                    self.plt.imshow(v)
            self.plt.show()

    def save_img(self, img_id, path):
        cv2.imwrite(os.path.join(path, "{}.jpg".format(img_id)), self.imgs[img_id])

    def get_img(self, img_id):
        return self.imgs[img_id]

    def save_all_imgs(self, image_name, path):
        for _, v in self.imgs.items():
            cv2.imwrite(os.path.join(path, image_name), v)

    @staticmethod
    def draw_polygons(image, polygons, color=(0, 0, 255), thickness=2):
        for polygon in polygons:
            polygon = np.array(polygon, dtype=np.int32)
            num_points = len(polygon) // 2
            for i in range(num_points):
                start_point = (polygon[2 * i], polygon[2 * i + 1])
                end_point = (polygon[2 * ((i + 1) % num_points)], polygon[2 * ((i + 1) % num_points) + 1])
                cv2.line(image, start_point, end_point, color, thickness)

    @staticmethod
    def draw_points(image, points, color=(0, 0, 255), thickness=2):
        for point in points:
            point = np.array(point[:2], dtype=np.int32)
            cv2.circle(image, (point[0], point[1]), thickness, color, thickness)

    @staticmethod
    def draw_logic_coords(image, logic_coords, points, gb_color=(219, 112, 147), font_color=(0, 0, 0), thickness=1):
        for i, logic_coord in enumerate(logic_coords):
            point = np.array(points[i][:2], dtype=np.int32)
            logic_coord = np.array(logic_coord, dtype=np.int32)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt = "{},{},{},{}".format(logic_coord[0], logic_coord[1], logic_coord[2], logic_coord[3])
            cat_size = cv2.getTextSize(txt, font, 0.3, 2)[0]
            cv2.rectangle(image, (point[0], point[1] - cat_size[1] - 2), (point[0] + cat_size[0], point[1] - 2), gb_color, -1)
            cv2.putText(image, txt, (point[0], point[1] - 2), font, 0.30, font_color, thickness=thickness, lineType=cv2.LINE_AA)
