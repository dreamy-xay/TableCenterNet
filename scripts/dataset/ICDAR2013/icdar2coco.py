#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-12-06 11:24:44
LastEditors: dreamy-xay
LastEditTime: 2024-12-30 11:08:32
'''
import os
import json
from tqdm import tqdm
import argparse
import cv2
import numpy as np

def padding_image(image, pad_value=None, pixel_value=0):
    if pad_value is None or pad_value == 0:
        return image

    if isinstance(pad_value, int):
        pad_value = (pad_value, pad_value, pad_value, pad_value)

    h, w = image.shape[:2]

    new_h = h + pad_value[0] + pad_value[2]
    new_w = w + pad_value[1] + pad_value[3]

    image_padded = np.zeros((new_h, new_w, 3), dtype=image.dtype)
    if pixel_value != 0:
        image_padded.fill(pixel_value)
    image_padded[pad_value[0] : pad_value[0] + h, pad_value[1] : pad_value[1] + w] = image

    return image_padded

class ICDAR2COCO:
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

    def parse_json_files(self, image_dir_path, json_dir_path, padding_image_margin=0):
        if padding_image_margin > 0:
            pim_imags_path = os.path.join(os.path.dirname(json_dir_path), ".." , f"padding{padding_image_margin}_images")
            if not os.path.exists(pim_imags_path):
                os.makedirs(pim_imags_path)
        
        # Get the list of json files
        json_file_paths = [json_file_path for json_file_path in os.listdir(json_dir_path) if json_file_path.endswith('.json') and os.path.isfile(os.path.join(json_dir_path, json_file_path))]
        
        current_category_id = self.add_cat_item("box")

        # Go through all files
        for json_file_path in tqdm(json_file_paths, desc="Parse Xml Files", unit="file"):
            current_image_id = None
            
            file_name = os.path.splitext(os.path.basename(json_file_path))[0] + '.jpg'

            json_file = os.path.join(json_dir_path, json_file_path)
            image_file = os.path.join(image_dir_path, file_name)
            
            if not os.path.exists(image_file):
                continue
            
            image = cv2.imread(image_file)
            if padding_image_margin > 0:
                image = padding_image(image, padding_image_margin)
                cv2.imwrite(os.path.join(pim_imags_path, file_name), image)
            
            height, width = image.shape[:2]
            size = {'width': width, 'height': height}
            
            current_image_id = self.add_image_item(file_name, size)
            
            with open(json_file, 'r') as f:
                data = json.load(f)

            for cell in data["cells"]:
                xmin, ymin, xmax, ymax = cell['xmin'], cell['ymin'], cell['xmax'], cell['ymax']
                if padding_image_margin > 0:
                    xmin += padding_image_margin
                    ymin += padding_image_margin
                    xmax += padding_image_margin
                    ymax += padding_image_margin
                
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
    arg_parser.add_argument("--input_json_dir", required=True, type=str)
    arg_parser.add_argument("--input_image_dir", required=True, type=str)
    arg_parser.add_argument("--output_json_path", required=True, type=str)
    arg_parser.add_argument("--vis_path", default="", type=str)
    arg_parser.add_argument("--pim", default=0, type=int)
    args = arg_parser.parse_args()
    
    coco_data = ICDAR2COCO().parse_json_files(args.input_image_dir, args.input_json_dir, args.pim).get_coco_data()
    
    with open(args.output_json_path, 'w') as f:
        json.dump(coco_data, f)
    
    if args.vis_path:
        images_dir = os.path.join(os.path.dirname(args.input_json_dir), ".." , f"padding{args.pim}_images") if args.pim > 0 else args.input_image_dir
        vis_result(args.output_json_path, images_dir, args.vis_path)
    

