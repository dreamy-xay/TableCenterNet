#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 15:35:22
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 15:36:00
"""
from engine.table.trainer import TableTrainer
from .loss import MTableLoss
from .dataset import get_dataset


class MTableTrainer(TableTrainer):
    def __init__(self, args):
        super().__init__(args, loss=MTableLoss((1, 1, 1, 1, 1)), dataset=get_dataset(args))
