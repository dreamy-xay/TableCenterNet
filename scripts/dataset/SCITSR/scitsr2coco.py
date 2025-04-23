#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-12-06 11:24:44
LastEditors: dreamy-xay
LastEditTime: 2025-01-03 19:51:42
"""
import os
import json
from tqdm import tqdm
import argparse
import cv2

try:
    import fitz
except:
    pass
from PIL import Image
import io
import numpy as np
import shutil


def pdf_to_images(pdf_path):
    # 打开 PDF 文件
    doc = fitz.open(pdf_path)

    # 用来存储所有页面的图像
    images = []

    # 遍历 PDF 中的每一页
    for page_num in range(len(doc)):
        page = doc[page_num]

        mat = fitz.Matrix(1, 1)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 将图像添加到列表中
        images.append(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR))

    return images


def resize_image(image, scale):
    """
    根据缩放因子调整图像大小
    :param image: 输入图像（OpenCV 格式）
    :param scale: 缩放因子（例如 0.5 表示缩小为原来的一半，2 表示放大为原来的两倍）
    :return: 调整大小后的图像
    """
    # 获取原始图像的尺寸
    height, width = image.shape[:2]

    # 计算新的尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 选择插值方法：放大时使用INTER_CUBIC，缩小时使用INTER_AREA
    if scale > 1:
        interpolation = cv2.INTER_CUBIC  # 放大时使用三次插值
    else:
        interpolation = cv2.INTER_AREA  # 缩小时使用区域插值（质量较好）

    # 调整图像大小
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)

    return resized_image


class SCITSR2COCO:
    def __init__(self):
        self.coco = {"images": [], "type": "instances", "annotations": [], "categories": []}

        self.category_set = dict()
        self.image_set = set()

        self.category_item_id = 0
        self.image_id = 20140000000
        self.annotation_id = 0

    def add_cat_item(self, name):
        if name in self.category_set:
            return self.category_set[name]

        self.category_item_id += 1
        self.category_set[name] = self.category_item_id

        category_item = {
            "supercategory": "none",
            "id": self.category_item_id,
            "name": name,
        }
        self.coco["categories"].append(category_item)

        return self.category_item_id

    def add_image_item(self, file_name, size):
        if file_name in self.image_set:
            raise Exception("duplicated image: {}".format(file_name))
        if file_name is None:
            raise Exception("Could not find filename tag in xml file.")
        if size["width"] is None:
            raise Exception("Could not find width tag in xml file.")
        if size["height"] is None:
            raise Exception("Could not find height tag in xml file.")

        self.image_id += 1
        self.image_set.add(file_name)

        image_item = {
            "id": self.image_id,
            "file_name": file_name,
            "width": size["width"],
            "height": size["height"],
        }
        self.coco["images"].append(image_item)

        return self.image_id

    def add_anno_item(self, image_id, category_id, bbox, seg, logic_axis):
        self.annotation_id += 1

        annotation_item = {
            "id": self.annotation_id,
            "category_id": category_id,
            "image_id": image_id,
            "bbox": bbox,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0,
            "ignore": 0,
            "segmentation": [seg],
            "logic_axis": [logic_axis],
        }

        self.coco["annotations"].append(annotation_item)

    def parse(self, data_dir, save_image_dir, mode="comp", pad=10):  # mode: ['comp', 'train', 'test']
        dir_name = "train" if mode == "train" else "test"

        pdf_dir = os.path.join(data_dir, dir_name, "pdf")
        chunk_dir = os.path.join(data_dir, dir_name, "chunk")
        structure_dir = os.path.join(data_dir, dir_name, "structure")

        # 获取文件名列表
        if mode == "comp":
            with open(os.path.join(data_dir, "SciTSR-COMP.list")) as f:
                file_name_list = f.read().splitlines()
        elif mode == "train" or mode == "test":
            file_name_list = [
                os.path.splitext(json_file_path)[0] for json_file_path in os.listdir(structure_dir) if json_file_path.endswith(".json") and os.path.isfile(os.path.join(structure_dir, json_file_path))
            ]
        else:
            raise Exception("Invalid mode: {}".format(mode))

        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)

        current_category_id = self.add_cat_item("box")

        # 遍历所有文件
        for file_name in tqdm(file_name_list, desc="Parse Xml Files", unit="file"):
            chunk_file = os.path.join(chunk_dir, file_name + ".chunk")
            structure_file = os.path.join(structure_dir, file_name + ".json")
            image_file = os.path.join(save_image_dir, file_name + ".jpg")
            pdf_file = os.path.join(pdf_dir, file_name + ".pdf")

            if not os.path.exists(pdf_file) or not os.path.exists(structure_file) or not os.path.exists(chunk_file):
                print(f"File not found: {file_name}")
                continue

            scale = 2.0
            image = pdf_to_images(pdf_file)[0]
            image = resize_image(image, 2.0)
            height, width = image.shape[:2]

            with open(chunk_file, "r") as f:
                chunk = json.load(f)

            cell_boxes = []
            for cell in chunk["chunks"]:
                pos = list(map(lambda x: x * scale, cell["pos"]))
                xmin = min(pos[:2])
                ymin = min(height - pos[2], height - pos[3])
                xmax = max(pos[:2])
                ymax = max(height - pos[2], height - pos[3])
                cell_boxes.append([xmin, ymin, xmax, ymax])

            min_x = max(0, min([int(box[0]) for box in cell_boxes]) - pad)
            min_y = max(0, min([int(box[1]) for box in cell_boxes]) - pad)
            max_x = min(width, max([int(box[2]) for box in cell_boxes]) + pad + 1)
            max_y = min(height, max([int(box[3]) for box in cell_boxes]) + pad + 1)

            image = image[min_y:max_y, min_x:max_x]
            cv2.imwrite(image_file, image)

            size = {"width": max_x - min_x, "height": max_y - min_y}

            current_image_id = self.add_image_item(file_name + ".jpg", size)

            for cell_box in cell_boxes:
                cell_box[0] -= min_x
                cell_box[1] -= min_y
                cell_box[2] -= min_x
                cell_box[3] -= min_y

            with open(structure_file, "r") as f:
                structure = json.load(f)

            sub_index = 0

            for cell in sorted(structure["cells"], key=lambda cell: cell["id"]):
                cell_id = cell["id"]

                if len(cell["tex"]) == 0 and len(cell["content"]) == 0:
                    sub_index += 1
                    continue

                cell_id = int(cell_id) - sub_index

                try:
                    xmin, ymin, xmax, ymax = cell_boxes[cell_id]
                except:
                    # raise Exception(f"Cell {cell_id}/{len(chunk['chunks'])} not found in chunk file {chunk_file}.(structure file: {structure_file})")
                    continue

                seg = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                logic_axis = [cell["start_col"], cell["end_col"], cell["start_row"], cell["end_row"]]
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                self.add_anno_item(current_image_id, current_category_id, bbox, seg, logic_axis)

        return self

    def parse_structure(self, data_dir, save_image_dir, mode="comp"):  # mode: ['comp', 'train', 'test']
        dir_name = "train" if mode == "train" else "test"

        image_dir = os.path.join(data_dir, dir_name, "img")
        structure_dir = os.path.join(data_dir, dir_name, "structure")

        # 获取文件名列表
        if mode == "comp":
            with open(os.path.join(data_dir, "SciTSR-COMP.list")) as f:
                file_name_list = f.read().splitlines()
        elif mode == "train" or mode == "test":
            file_name_list = [
                os.path.splitext(json_file_path)[0] for json_file_path in os.listdir(structure_dir) if json_file_path.endswith(".json") and os.path.isfile(os.path.join(structure_dir, json_file_path))
            ]
        else:
            raise Exception("Invalid mode: {}".format(mode))

        if not os.path.exists(save_image_dir):
            os.makedirs(save_image_dir)

        current_category_id = self.add_cat_item("box")

        # 遍历所有文件
        for file_name in tqdm(file_name_list, desc="Parse Xml Files", unit="file"):
            structure_file = os.path.join(structure_dir, file_name + ".json")
            image_file = os.path.join(image_dir, file_name + ".png")
            save_image_file = os.path.join(save_image_dir, file_name + ".png")

            if not os.path.exists(image_file) or not os.path.exists(structure_file):
                print(f"File not found: {file_name}")
                continue

            shutil.copyfile(image_file, save_image_file)
            image = cv2.imread(image_file)

            size = {"width": image.shape[1], "height": image.shape[0]}

            current_image_id = self.add_image_item(file_name + ".png", size)

            with open(structure_file, "r") as f:
                structure = json.load(f)

            for cell in sorted(structure["cells"], key=lambda cell: cell["id"]):
                seg = [0, 0, 0, 0, 0, 0, 0, 0]
                logic_axis = [cell["start_col"], cell["end_col"], cell["start_row"], cell["end_row"]]
                bbox = [0, 0, 0, 0]

                self.add_anno_item(current_image_id, current_category_id, bbox, seg, logic_axis)

        return self

    def get_coco_data(self):
        return self.coco


def vis_result(coco_json, images_dir, shows_dir):
    if os.path.exists(shows_dir):
        os.system("rm -rf {}".format(shows_dir))
    os.makedirs(shows_dir)

    from pycocotools.coco import COCO

    coco = COCO(coco_json)

    for img_id in coco.getImgIds():
        img = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img["id"])
        anns = coco.loadAnns(ann_ids)

        image = cv2.imread(os.path.join(images_dir, img["file_name"]))

        for ann in anns:
            bbox = ann["bbox"]
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 1)

        cv2.imwrite(os.path.join(shows_dir, img["file_name"]), image)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", required=True, type=str)
    arg_parser.add_argument("--save_image_dir", required=True, type=str)
    arg_parser.add_argument("--output_json_path", required=True, type=str)
    arg_parser.add_argument("--mode", default="comp", type=str)
    arg_parser.add_argument("--only_structure", action="store_true")
    arg_parser.add_argument("--vis_path", default="", type=str)
    args = arg_parser.parse_args()

    if args.only_structure:
        coco_data = SCITSR2COCO().parse_structure(args.data_dir, args.save_image_dir, args.mode).get_coco_data()
    else:
        coco_data = SCITSR2COCO().parse(args.data_dir, args.save_image_dir, args.mode).get_coco_data()

    with open(args.output_json_path, "w") as f:
        json.dump(coco_data, f)

    if args.vis_path:
        vis_result(args.output_json_path, args.save_image_dir, args.vis_path)
