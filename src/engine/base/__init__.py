#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-24 11:05:45
LastEditors: dreamy-xay
LastEditTime: 2024-10-25 15:14:26
'''
from .argparser import BaseArgParser
from .trainer import BaseTrainer
from .validator import BaseValidator
from .predictor import BasePredictor


def get_engine():
    return BaseArgParser, BaseTrainer, BaseValidator, BasePredictor