#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
Version:
Autor: dreamy-xay
Date: 2024-10-22 11:16:02
LastEditors: dreamy-xay
LastEditTime: 2024-11-16 14:22:24
"""
import importlib
import sys
import os


def main(task):
    ArgParser, Trainer, Validator, Predictor = importlib.import_module(f"engine.{task}").get_engine()

    args = ArgParser().parse()

    if args.task == "base":
        raise ValueError("task must be specified, task cannot be a 'base'.")

    if args.mode == "train":
        trainer = Trainer(args)
        trainer.run()
    elif args.mode == "val":
        validator = Validator(args)
        validator.run()
    elif args.mode == "predict":
        predictor = Predictor(args)
        predictor.run()
    else:
        raise ValueError("mode must be 'train' or 'val' or 'predict'")


if __name__ == "__main__":
    # Main function
    main(sys.argv[1] if len(sys.argv) >= 2 and not sys.argv[1].startswith("-") else "base")
