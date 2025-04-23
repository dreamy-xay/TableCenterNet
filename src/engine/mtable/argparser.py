#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 10:35:11
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 10:42:26
"""
from engine.table.argparser import TableArgParser


class MTableArgParser(TableArgParser):

    def __init__(self):
        super().__init__()

    def add_predict_args(self, parser):
        super().add_predict_args(parser)
        parser.add_argument("--not_relocate", action="store_true", help="Whether to not reposition the cell corners?")

    def add_val_args(self, parser):
        super().add_val_args(parser)
        parser.add_argument("--not_relocate", action="store_true", help="Whether to not reposition the cell corners?")
