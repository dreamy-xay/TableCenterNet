#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-12-06 11:24:44
LastEditors: dreamy-xay
LastEditTime: 2024-12-28 16:54:19
"""
import os
import json
from tqdm import tqdm
import argparse
import cv2
import xml.etree.ElementTree as ET
import numpy as np


def sort_points_clockwise(points: np.ndarray) -> np.ndarray:
    """
    Sorts the given points in clockwise order with the point closest to the top-left corner (minimum bounding rectangle's top-left corner) as the first point.

    Args:
        points (ndarray): numpy array of shape (n, 2) containing the coordinates of n points.

    Returns:
        sorted_points (ndarray): numpy array of shape (n, 2) containing the sorted coordinates of the points.
    """
    # 计算每个点到原点的距离
    distances = np.sum(points**2, axis=1)

    # 找到距离最小的点作为起始点
    start_index = np.argmin(distances)

    # 起始第一个点
    start_point = points[start_index]

    # 将起始点放在列表的第一个位置
    sorted_points = np.array([start_point])
    remaining_points = np.concatenate((points[:start_index], points[start_index + 1 :]), axis=0)

    # 计算每个点相对于起始点的极角，并按照极角排序
    remaining_points = sorted(remaining_points, key=lambda p: np.arctan2(p[1] - start_point[1], p[0] - start_point[0]))

    # 将排序后的点添加到结果列表中
    sorted_points = np.concatenate((sorted_points, remaining_points), axis=0)

    return sorted_points

def xml_to_dict(element):
    # 递归转换XML元素为字典
    if len(element) == 0:  # 如果没有子元素
        return element.attrib

    # 如果有子元素，则将它们转化为字典
    result = {}
    for child in element:
        child_dict = xml_to_dict(child)
        if child.tag in result:
            # 如果tag重复，转换成列表
            if isinstance(result[child.tag], list):
                result[child.tag].append(child_dict)
            else:
                result[child.tag] = [result[child.tag], child_dict]
        else:
            result[child.tag] = child_dict

    # 将属性添加到字典中
    if element.attrib:
        result.update(element.attrib)

    return result


class ICDARM2COCO:
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

    def parse_xml_files(self, image_dir_path, xml_dir_path, output_image_dir, pad):
        # 获取xml文件列表
        xml_file_paths = [xml_file_path for xml_file_path in os.listdir(xml_dir_path) if xml_file_path.endswith(".xml") and os.path.isfile(os.path.join(xml_dir_path, xml_file_path))]

        current_category_id = self.add_cat_item("box")

        # 遍历所有文件
        for xml_file_path in tqdm(xml_file_paths, desc="Parse Xml Files", unit="file"):
            xml_file = os.path.join(xml_dir_path, xml_file_path)

            tree = ET.parse(xml_file)  # 解析 XML 文件
            root = tree.getroot()  # 获取根元素
            data = xml_to_dict(root)  # 转换为字典

            file_name = data["filename"]
            image_name = os.path.splitext(file_name)[0]
            image_file = os.path.join(image_dir_path, file_name)

            if not os.path.exists(image_file):
                print(f"Image file {image_file} not found.")
                continue

            image = cv2.imread(image_file)
            
            for table_id, table in enumerate((data["table"] if isinstance(data["table"], list) else [data["table"]]) if "table" in data else []):
                xml_cells = table["cell"] if isinstance(table["cell"], list) else [table["cell"]]
                # table_image, offset = self._crop_table(image, pad, table)
                table_image, offset = self._crop_table_by_cells(image, pad, [self._parse_points(cell) for cell in xml_cells])
                
                table_image_name = f"{image_name}_{table_id}.jpg"
                
                size = {"width": table_image.shape[1], "height": table_image.shape[0]}

                current_image_id = self.add_image_item(table_image_name, size)
                cv2.imwrite(os.path.join(output_image_dir, table_image_name), table_image)
                
                for cell in xml_cells:
                    points = self._parse_points(cell, offset, True)

                    if len(points) != 8:
                        continue

                    xmin = min(points[0::2])
                    ymin = min(points[1::2])
                    xmax = max(points[0::2])
                    ymax = max(points[1::2])

                    seg = points
                    logic_axis = list(map(int, [cell["start-col"], cell["end-col"], cell["start-row"], cell["end-row"]]))
                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                    self.add_anno_item(current_image_id, current_category_id, bbox, seg, logic_axis)

        return self

    @staticmethod
    def _parse_points(cell, offset=[0, 0], sort_points=False):
        points = cell["Coords"]["points"].split()
        for i in range(len(points)):
            x, y = points[i].split(",")
            points[i] = [int(x) - offset[0], int(y) - offset[1]]

        # 原始点按顺时针排序
        if sort_points:
            points = sort_points_clockwise(np.array(points)).reshape(-1).tolist()
        else:
            points = [coord for point in points for coord in point]

        return points

    @staticmethod
    def _crop_table(image, pad, table):
        points = table["Coords"]["points"]
        points = [int(coord) for point in points.split() for coord in point.split(",")]
        
        height, width = image.shape[:2]

        x_min = max(min(points[0::2]) - pad, 0)
        y_min = max(min(points[1::2]) - pad, 0)
        x_max = min(max(points[0::2]) + pad + 1, width)
        y_max = min(max(points[1::2]) + pad + 1, height)

        return image[y_min:y_max, x_min:x_max], (x_min, y_min)

    @staticmethod
    def _crop_table_by_cells(image, pad, cells):
        height, width = image.shape[:2]

        x_min = max(min(min(points[0::2]) for points in cells) - pad, 0)
        y_min = max(min(min(points[1::2]) for points in cells) - pad, 0)
        x_max = min(max(max(points[0::2]) for points in cells) + pad + 1, width)
        y_max = min(max(max(points[1::2]) for points in cells) + pad + 1, height)

        return image[y_min:y_max, x_min:x_max], (x_min, y_min)

    def get_coco_data(self):
        return self.coco


def vis_result(coco_json, images_dir, shows_dir):
    if os.path.exists(shows_dir):
        os.system('rm -rf {}'.format(shows_dir))
    os.makedirs(shows_dir)
    
    from pycocotools.coco import COCO

    coco = COCO(coco_json)

    for img_id in coco.getImgIds():
        img = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img['id'])
        anns = coco.loadAnns(ann_ids)
        
        image = cv2.imread(os.path.join(images_dir, img['file_name']))
        
        for ann in anns:
            bbox = ann['bbox']
            cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 1)
        
        cv2.imwrite(os.path.join(shows_dir, img['file_name']), image)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_xml_dir", required=True, type=str)
    arg_parser.add_argument("--input_image_dir", required=True, type=str)
    arg_parser.add_argument("--output_image_dir", required=True, type=str)
    arg_parser.add_argument("--output_json_path", required=True, type=str)
    arg_parser.add_argument("--pad", default=4, type=int)
    arg_parser.add_argument("--vis_path", default="", type=str)
    args = arg_parser.parse_args()

    coco_data = ICDARM2COCO().parse_xml_files(args.input_image_dir, args.input_xml_dir, args.output_image_dir, args.pad).get_coco_data()

    with open(args.output_json_path, "w") as f:
        json.dump(coco_data, f)
    
    if args.vis_path:
        vis_result(args.output_json_path, args.output_image_dir, args.vis_path)
