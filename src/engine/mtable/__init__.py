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
from .argparser import MTableArgParser
from .trainer import MTableTrainer
from .validator import MTableValidator
from .predictor import MTablePredictor


def get_engine():
    return MTableArgParser, MTableTrainer, MTableValidator, MTablePredictor
