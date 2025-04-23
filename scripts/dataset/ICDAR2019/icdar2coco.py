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
from typing import List, Tuple


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


class QuaqFitter:
    """Quadrilateral fitter."""

    def __init__(self, contours: List[np.ndarray]):
        """
        Initialization.

        Args:
            contours (list): Contours.
        """
        self.contours = []

        # 初始化数据
        for contour in contours:
            self.contours.append(contour if isinstance(contour, np.ndarray) else np.asarray(contour, dtype=np.float32))

    def __call__(self) -> List[np.ndarray]:
        """
        Fitting contour data is called quadrilateral data.

        Returns:
            polygons (List): Returns the fitted polygon. The final result of the fitting is a quadrilateral, but it does not exclude the existence of polygons with less than four sides.
        """
        polygons = []

        for contour in self.contours:
            # 拟合成多边形
            polygon = contour

            # 如果大于四边形，则计算凸包，删除无效点
            if len(polygon) > 4:
                polygon = cv2.convexHull(polygon, returnPoints=True).reshape(-1, 2)
                # print("ConvexHull:\n", polygon)

            # 如果大于四边形，则计算凸多边形的最大内接四边形
            if len(polygon) > 4:
                polygon = self.polygon2quad(polygon)
                # print("Polygon2Quad:\n", polygon)

            polygons.append(polygon)

        return polygons

    @staticmethod
    def polygon2quad(points: np.ndarray) -> np.ndarray:
        """
        Convert a polygon to a quadrilateral.

        Args:
            points (ndarray): Polygon points.

        Returns:
            points (ndarray): Returns the quadrilateral points.
        """
        # 将凸多边形的顶点坐标转换为numpy数组
        if not isinstance(points, np.ndarray):
            points = np.asarray(points)

        # 获取凸多边形的聚类结果
        clusters = QuaqFitter.kmeans(points, 4, 10)

        # 初始化最大内接四边形的顶点坐标
        max_quad_area = 0
        max_quad = np.empty((0, 2))

        # 遍历所有簇，寻找最大内接四边形
        for cluster0 in clusters[0]:
            for cluster1 in clusters[1]:
                for cluster2 in clusters[2]:
                    for cluster3 in clusters[3]:
                        # 按顺时针排序
                        quad = [cluster0, cluster1, cluster2, cluster3]
                        quad.sort(key=lambda cluster: cluster[0])

                        # 抽离出点坐标
                        quad_contour = np.asarray(list(map(lambda cluster: cluster[1], quad)))

                        # 计算四边形面积
                        quad_area = cv2.contourArea(quad_contour)

                        if quad_area > max_quad_area:
                            max_quad_area = quad_area
                            max_quad = quad_contour

        return max_quad

    @staticmethod
    def kmeans(points: np.ndarray, k: int = 4, attempts: int = 10) -> List[List[Tuple[int, np.ndarray]]]:
        """
        K-means clustering of list of points.

        Args:
            points (ndarray): List of points.
            k (int): Number of clusters.
            attempts (int): Number of attempts.

        Returns:
            clusters (List[List[(int, np.ndarray)]]): Returns the clustered points and their index in the original point list.
        """
        # 设置 K 均值聚类的参数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

        # 执行 K 均值聚类
        _, labels, __ = cv2.kmeans(points, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

        # 根据聚类结果对数据点进行分组
        clusters: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            clusters[label[0]].append((i, points[i]))

        return clusters

    @staticmethod
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


class ICDAR2COCO:
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

    def add_anno_item(self, image_id, category_id, bbox, seg, logic_axis, table_id):
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
            "table_id": table_id,
        }

        self.coco["annotations"].append(annotation_item)

    def parse_xml_files(self, image_dir_path, xml_dir_path):
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
            image_file = os.path.join(image_dir_path, file_name)

            if not os.path.exists(image_file):
                print(f"Image file {image_file} not found.")
                continue

            image = cv2.imread(image_file)
            height, width = image.shape[:2]
            size = {"width": width, "height": height}

            current_image_id = self.add_image_item(file_name, size)

            for table_id, table in enumerate((data["table"] if isinstance(data["table"], list) else [data["table"]]) if "table" in data else []):
                for cell in table["cell"] if isinstance(table["cell"], list) else [table["cell"]]:
                    points = self._parse_points(cell["Coords"]["points"])

                    if len(points) != 8:
                        continue

                    xmin = min(points[0], points[2], points[4], points[6])
                    ymin = min(points[1], points[3], points[5], points[7])
                    xmax = max(points[0], points[2], points[4], points[6])
                    ymax = max(points[1], points[3], points[5], points[7])

                    seg = points
                    logic_axis = list(map(int, [cell["start-col"], cell["end-col"], cell["start-row"], cell["end-row"]]))
                    bbox = [xmin, ymin, xmax - xmin, ymax - ymin]

                    self.add_anno_item(current_image_id, current_category_id, bbox, seg, logic_axis, table_id)

        return self

    @staticmethod
    def _parse_points(points_str):
        points = points_str.split()
        for i in range(len(points)):
            x, y = points[i].split(",")
            points[i] = [int(x), int(y)]

        if len(points) > 4:
            # 拟合四边形
            points = QuaqFitter([points])()[0]
            # print(9999)

        # 原始点按顺时针排序
        points = QuaqFitter.sort_points_clockwise(np.asarray(points)).reshape(-1).tolist()

        return points

    def get_coco_data(self):
        return self.coco


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input_xml_dir", required=True, type=str)
    arg_parser.add_argument("--input_image_dir", required=True, type=str)
    arg_parser.add_argument("--output_json_path", required=True, type=str)
    args = arg_parser.parse_args()

    coco_data = ICDAR2COCO().parse_xml_files(args.input_image_dir, args.input_xml_dir).get_coco_data()

    with open(args.output_json_path, "w") as f:
        json.dump(coco_data, f)
