#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 16:45:27
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 17:00:26
"""
import torch
import yaml
from models import *

GLOBALS = globals()


def create_model(model_yaml):
    with open(model_yaml, "r", encoding="utf-8") as f:
        model_info = yaml.safe_load(f)  # Load the YAML content as a dictionary

        if model_info["name"] not in GLOBALS:
            raise ValueError(f"Unknown function name: {model_info['name']}")

        model = GLOBALS[model_info["name"]](**model_info["params"])

    return model


def load_model(model, model_path, optimizer=None, resume=False, lr=None, lr_step=None):
    start_epoch = 0

    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)

    print("loaded {}, epoch {}".format(model_path, checkpoint["epoch"]))
    state_dict_ = checkpoint["state_dict"]
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print("Skip loading parameter {}, required shape{}, " "loaded shape{}.".format(k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print("Drop parameter {}.".format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr
            print("Resumed optimizer with start lr", start_lr)
        else:
            print("No optimizer parameters in checkpoint.")
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {"epoch": epoch, "state_dict": state_dict}
    if not (optimizer is None):
        data["optimizer"] = optimizer.state_dict()
    torch.save(data, path)
