#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-24 11:05:45
LastEditors: dreamy-xay
LastEditTime: 2024-11-05 14:30:01
"""
from .argparser import STableArgParser
from .trainer import STableTrainer
from .validator import STableValidator
from .predictor import STablePredictor


def get_engine():
    return STableArgParser, STableTrainer, STableValidator, STablePredictor
