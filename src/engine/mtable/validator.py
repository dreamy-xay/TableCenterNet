#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 10:34:01
LastEditors: dreamy-xay
LastEditTime: 2024-10-24 11:20:23
"""
from engine.table.validator import TableValidator
from .predictor import MTablePredictor


class MTableValidator(TableValidator):
    def __init__(self, args):
        super().__init__(args, predictor=MTablePredictor)
