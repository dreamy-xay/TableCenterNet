#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-12-06 11:24:44
LastEditors: dreamy-xay
LastEditTime: 2024-12-06 12:27:34
'''
import xml.etree.ElementTree as ET
import os
import json
from tqdm import tqdm
import argparse

class WTW2COCO:
    def __init__(self):        
        self.coco = {'images': [], 'type': 'instances', 'annotations': [], 'categories': []}

        self.category_set = dict()
        self.image_set = set()

        self.category_item_id = 0
        self.image_id = 20140000000
        self.annotation_id = 0

    def add_cat_item(self, name):
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

    def add_anno_item(self, image_id, category_id, bbox, seg, logic_axis, table_id):
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
            'table_id': table_id,
        }
                
        self.coco['annotations'].append(annotation_item)

    def parse_xml_files(self, xml_dir_path):
        # 获取xml文件列表
        xml_file_paths = [xml_file_path for xml_file_path in os.listdir(xml_dir_path) if xml_file_path.endswith('.xml') and os.path.isfile(os.path.join(xml_dir_path, xml_file_path))]

        # 遍历所有文件
        for xml_file_path in tqdm(xml_file_paths, desc="Parse Xml Files", unit="file"):
            size = {'width': None, 'height': None, 'depth': None}
            current_image_id = None
            current_category_id = None
            file_name = None

            xml_file = os.path.join(xml_dir_path, xml_file_path)

            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            if root.tag != 'annotation':
                raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

            # elem is <folder>, <filename>, <size>, <object>
            for elem in root:
                current_parent = elem.tag
                object_name = None
                
                if elem.tag == 'folder':
                    continue
                
                if elem.tag == 'filename':
                    file_name = elem.text
                    if file_name in self.category_set:
                        raise Exception('file_name duplicated')
                # add img item only after parse <size> tag
                elif current_image_id is None and file_name is not None and size['width'] is not None:
                    if file_name not in self.image_set:
                        current_image_id = self.add_image_item(file_name, size)
                    else:
                        raise Exception('duplicated image: {}'.format(file_name)) 
                
                # subelem is <width>, <height>, <depth>, <name>, <bndbox>
                for subelem in elem:
                    if current_parent == 'object' and subelem.tag == 'name':
                        object_name = subelem.text
                        if object_name not in self.category_set:
                            current_category_id = self.add_cat_item(object_name)
                        else:
                            current_category_id = self.category_set[object_name]
                    elif current_parent == 'size':
                        if size[subelem.tag] is not None:
                            raise Exception('xml structure broken at size tag.')
                        size[subelem.tag] = int(subelem.text)
                    elif subelem.tag == 'bndbox':
                        if current_image_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        if current_category_id is None:
                            raise Exception('xml structure broken at bndbox tag')
                        
                        # option is <xmin>, <ymin>, <xmax>, <ymax>, <x1>~<y4>, <startcol>~<endrow>, <tableid>, when subelem is <bndbox>
                        bndbox = {}
                        
                        for option in subelem:
                            bndbox[option.tag] = float(option.text)
                        
                        seg = [bndbox['x1'], bndbox['y1'], bndbox['x2'], bndbox['y2'], bndbox['x3'], bndbox['y3'], bndbox['x4'], bndbox['y4']]
                        logic_axis = list(map(int, [bndbox['startcol'], bndbox['endcol'], bndbox['startrow'], bndbox['endrow']]))
                        table_id = int(bndbox['tableid'])
        
                        bbox = [bndbox['xmin'], bndbox['ymin'], bndbox['xmax'] - bndbox['xmin'], bndbox['ymax'] - bndbox['ymin']]
                        self.add_anno_item(current_image_id, current_category_id, bbox, seg, logic_axis, table_id)
        
        return self

    def get_coco_data(self):
        return self.coco

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--xml_dir", required=True, type=str)
    arg_parser.add_argument("--json_path", required=True, type=str)
    args = arg_parser.parse_args()
    
    coco_data = WTW2COCO().parse_xml_files(args.xml_dir).get_coco_data()
    
    with open(args.json_path, 'w') as f:
        json.dump(coco_data, f)
    

