#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-12-06 11:24:44
LastEditors: dreamy-xay
LastEditTime: 2024-12-31 17:00:21
'''
import os
import json
from tqdm import tqdm
import argparse
import cv2

class SCITSR2COCO:
    def __init__(self):        
        self.coco = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}

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
            'supercategory': 'none',
            'id': self.category_item_id,
            'name': name,
        }
        self.coco['categories'].append(category_item)
        
        return self.category_item_id

    def add_image_item(self, file_name, size):
        if file_name in self.image_set:
            raise Exception('duplicated image: {}'.format(file_name))
        if file_name is None:
            raise Exception('Could not find filename tag in xml file.')
        if size['width'] is None:
            raise Exception('Could not find width tag in xml file.')
        if size['height'] is None:
            raise Exception('Could not find height tag in xml file.')
        
        self.image_id += 1
        self.image_set.add(file_name)
        
        image_item = {
            'id': self.image_id,
            'file_name': file_name,
            'width': size['width'],
            'height': size['height'],
        }
        self.coco['images'].append(image_item)
        
        return self.image_id

    def add_anno_item(self, image_id, category_id, bbox, seg, logic_axis):
        self.annotation_id += 1
        
        annotation_item = {
            'id': self.annotation_id,
            'category_id': category_id,
            'image_id': image_id,
            'bbox': bbox,
            'area': bbox[2] * bbox[3],
            'iscrowd': 0,
            'ignore': 0,
            'segmentation': [seg],
            'logic_axis': [logic_axis],
        }
                
        self.coco['annotations'].append(annotation_item)

    def parse(self, data_dir, mode="comp"): # mode: ['comp', 'train', 'test']
        dir_name = "train" if mode == "train" else "test"
        
        image_dir = os.path.join(data_dir, dir_name, 'img')
        chunk_dir = os.path.join(data_dir, dir_name, 'chunk')
        structure_dir = os.path.join(data_dir, dir_name, 'structure')
        
        # 获取文件名列表
        if mode == "comp":
            with open(os.path.join(data_dir, 'SciTSR-COMP.list')) as f:
                file_name_list = f.read().splitlines()
        elif mode == "train" or mode == "test":
            file_name_list = [os.path.splitext(json_file_path)[0] for json_file_path in os.listdir(structure_dir) if json_file_path.endswith('.json') and os.path.isfile(os.path.join(structure_dir, json_file_path))]
        else:
            raise Exception("Invalid mode: {}".format(mode))
        
        current_category_id = self.add_cat_item("box")

        # 遍历所有文件
        for file_name in tqdm(file_name_list, desc="Parse Xml Files", unit="file"):
            chunk_file = os.path.join(chunk_dir, file_name + '.chunk')
            structure_file = os.path.join(structure_dir, file_name + '.json')
            image_file = os.path.join(image_dir, file_name + '.png')
            
            if not os.path.exists(image_file) or not os.path.exists(structure_file) or not os.path.exists(chunk_file):
                print(f"File not found: {file_name}")
                continue
            
            image = cv2.imread(image_file)
            height, width = image.shape[:2]
            size = {'width': width, 'height': height}
            
            current_image_id = self.add_image_item(file_name + ".png", size)
            
            with open(structure_file, 'r') as f:
                structure = json.load(f)
            with open(chunk_file, 'r') as f:
                chunk = json.load(f)

            for cell in structure["cells"]:
                cell_id = cell["id"]
                if cell_id == 0:
                    continue
                cell_id = int(cell_id) - 1
                
                try:
                    pos = chunk["chunks"][cell_id]["pos"]
                except:
                    raise Exception(f"Cell {cell_id} not found in chunk file {chunk_file}")
                xmin = min(pos[0::2])
                ymin = min(pos[1::2])
                xmax = max(pos[0::2])
                ymax = max(pos[1::2])
                
                seg = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                logic_axis = [cell['start_col'], cell['end_col'], cell['start_row'], cell['end_row']]
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                
                self.add_anno_item(current_image_id, current_category_id, bbox, seg, logic_axis)
        
        return self

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

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_dir", required=True, type=str)
    arg_parser.add_argument("--output_json_path", required=True, type=str)
    arg_parser.add_argument("--mode", default="comp", type=str)
    arg_parser.add_argument("--vis_path", default="", type=str)
    args = arg_parser.parse_args()
    
    coco_data = SCITSR2COCO().parse(args.data_dir, args.mode).get_coco_data()
    
    with open(args.output_json_path, 'w') as f:
        json.dump(coco_data, f)
    
    if args.vis_path:
        vis_result(args.output_json_path, os.path.join(args.data_dir, 'img'), args.vis_path)
    

