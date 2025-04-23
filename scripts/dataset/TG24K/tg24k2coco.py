#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-12-06 11:24:44
LastEditors: dreamy-xay
LastEditTime: 2025-01-03 15:15:07
"""
import os
import json
from tqdm import tqdm
import argparse
import cv2
from csv import DictReader
import pickle


class TG24K2COCO:
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

    def parse(self, data_dir, mode="test"):  # mode: ['val', 'train', 'test']
        image_dir = os.path.join(data_dir, "image")
        coord_dir = os.path.join(data_dir, "graph_node")
        structure_dir = os.path.join(data_dir, "graph_target")

        # 获取文件名列表
        with open(os.path.join(data_dir, f"{mode}.txt")) as f:
            file_name_list = f.read().splitlines()
            file_name_list = [os.path.splitext(os.path.basename(file_name.split()[0]))[0] for file_name in file_name_list]

        current_category_id = self.add_cat_item("box")

        # 遍历所有文件
        for file_name in tqdm(file_name_list, desc="Parse Xml Files", unit="file"):
            coord_file = os.path.join(coord_dir, file_name + "_node.csv")
            structure_file = os.path.join(structure_dir, file_name + "_target.csv")
            image_file = os.path.join(image_dir, file_name + "_org.png")

            if not os.path.exists(image_file) or not os.path.exists(structure_file) or not os.path.exists(coord_file):
                print(f"File not found: {file_name}")
                continue

            image = cv2.imread(image_file)
            height, width = image.shape[:2]
            size = {"width": width, "height": height}

            current_image_id = self.add_image_item(file_name + "_org.png", size)

            cells = []
            with open(coord_file, "r") as f:
                for cell in DictReader(f):
                    cells.append(cell)

            cells_structure = []
            with open(structure_file, "r") as f:
                for cell in DictReader(f):
                    cells_structure.append(cell)

            for ind, cell in enumerate(cells):
                cs = cells_structure[ind]

                if cell["ind"] != cs["ind"]:
                    raise Exception("ind not match")

                seg = list(map(int, [cell["x1"], cell["y1"], cell["x2"], cell["y2"], cell["x3"], cell["y3"], cell["x4"], cell["y4"]]))

                xmin = min(seg[::2])
                ymin = min(seg[1::2])
                xmax = max(seg[::2])
                ymax = max(seg[1::2])

                logic_axis = list(map(int, [cs["start-col"], cs["end-col"], cs["start-row"], cs["end-row"]]))
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                self.add_anno_item(current_image_id, current_category_id, bbox, seg, logic_axis)

        return self

    def parse_pkl(self, data_dir, mode="test"):  # mode: ['val', 'train', 'test']
        image_dir = os.path.join(data_dir, "image")
        label_dir = os.path.join(data_dir, "gt")

        # 获取文件名列表
        with open(os.path.join(data_dir, f"{mode}.txt")) as f:
            file_name_list = f.read().splitlines()
            file_name_list = [os.path.basename(file_name.split()[0]) for file_name in file_name_list]

        current_category_id = self.add_cat_item("box")

        # 遍历所有文件
        for file_name in tqdm(file_name_list, desc="Parse Gt Files", unit="file"):
            label_file = os.path.join(label_dir, file_name)

            with open(label_file, "rb") as f:
                data = pickle.load(f)

            image_name = os.path.basename(data["image_path"])
            image_file = os.path.join(image_dir, image_name)

            if not os.path.exists(image_file):
                print(f"File not found: {image_file}")
                continue

            image = cv2.imread(image_file)
            height, width = image.shape[:2]
            size = {"width": width, "height": height}

            current_image_id = self.add_image_item(image_name, size)

            for cell in data["cells_anno"]:
                seg = list(map(int, cell["bbox"]))

                xmin = min(seg[::2])
                ymin = min(seg[1::2])
                xmax = max(seg[::2])
                ymax = max(seg[1::2])

                lloc = cell["lloc"]
                logic_axis = list(map(int, [lloc[2], lloc[3], lloc[0], lloc[1]]))
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

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
    arg_parser.add_argument("--output_json_path", required=True, type=str)
    arg_parser.add_argument("--mode", default="test", type=str)
    arg_parser.add_argument("--vis_path", default="", type=str)
    arg_parser.add_argument("--pkl", action="store_true")
    args = arg_parser.parse_args()

    # 解析数据
    if args.pkl:
        coco_data = TG24K2COCO().parse_pkl(args.data_dir, args.mode).get_coco_data()
    else:
        coco_data = TG24K2COCO().parse(args.data_dir, args.mode).get_coco_data()

    # 保存 json 文件
    output_json_dir = os.path.dirname(args.output_json_path)
    if not os.path.exists(output_json_dir):
        os.makedirs(os.path.dirname(args.output_json_path), exist_ok=True)
    with open(args.output_json_path, "w") as f:
        json.dump(coco_data, f)

    # 可视化结果
    if args.vis_path:
        vis_result(args.output_json_path, os.path.join(args.data_dir, "image"), args.vis_path)
