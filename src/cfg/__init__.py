#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-23 15:51:31
LastEditors: dreamy-xay
LastEditTime: 2024-10-23 15:51:37
"""
import yaml
import os


def _load_config(path):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
DEFAULT_CFG = _load_config(os.path.join(os.path.dirname(__file__), "default.yaml"))
