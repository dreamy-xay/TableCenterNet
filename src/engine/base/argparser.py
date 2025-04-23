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
import argparse
import os
import shutil
from multiprocessing import cpu_count
from cfg import DEFAULT_CFG, ROOT_DIR, SRC_DIR


class BaseArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # *** task
        self.parser.add_argument("task", type=str, help="Tasks that need to be run.", choices=self._get_task_choices())

        # *** 创建子解析器
        subparsers = self.parser.add_subparsers(dest="mode", required=True, help="The running mode of the task, currently including 'train', 'val', 'predict' three modes.")

        # *** mode train, val, predict
        train_parser = subparsers.add_parser("train", help="Training mode.")
        val_parser = subparsers.add_parser("val", help="Validation mode.")
        predict_parser = subparsers.add_parser("predict", help="Prediction mode.")

        # *** 公共参数
        for mode_parser in [train_parser, val_parser, predict_parser]:
            mode_parser.add_argument("--project", type=str, default=None, help="Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.")
            mode_parser.add_argument(
                "--name", type=str, default=None, help="Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored."
            )
            mode_parser.add_argument(
                "--exist_ok",
                action="store_true",
                help="If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation without needing to manually clear previous outputs.",
            )
            mode_parser.add_argument(
                "--model",
                **(dict(required=True) if mode_parser != val_parser else dict(default="")),
                help="Specifies the model file for training. Accepts a path to either a .yaml configuration file. Essential for defining the model structure or initializing weights.",
            )
            self.add_public_args(mode_parser)

        # *** 训练参数
        train_parser.add_argument(
            "--data",
            required=True,
            help="Path to the dataset configuration file (e.g., coco8.yaml). This file contains dataset-specific parameters, including paths to training and validation data, class names, and number of classes.",
        )
        train_parser.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.",
        )
        train_parser.add_argument("--master_batch", type=int, default=-1, help="batch size on the master gpu.")
        train_parser.add_argument("--batch", type=int, default=32, help="Batch size, with three modes: set as an integer (e.g., batch=32).")
        train_parser.add_argument(
            "--save_period",
            type=int,
            default=10,
            help="Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.",
        )
        train_parser.add_argument(
            "--device",
            type=str,
            default="0",
            help="Specifies the computational device(s) for training: a single GPU (device=0), multiple GPUs (device=0,1).",
        )
        train_parser.add_argument(
            "--workers",
            type=int,
            default=8,
            help="Number of worker threads for data loading (per RANK if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.",
        )
        train_parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process.",
        )
        train_parser.add_argument("--model_path", type=str, default="", help="The path to a model that is retrained based on an already trained model.")
        train_parser.add_argument(
            "--resume",
            action="store_true",
            help="Resumes training from the last saved checkpoint. Automatically loads model weights, optimizer state, and epoch count, continuing training seamlessly.",
        )

        # *** 验证参数
        val_parser.add_argument("--label", type=str, required=True, help="The path where the tag corresponding to the source file is located, which can be a file or a directory.")
        val_parser.add_argument("--only_eval", action="store_true", help="Whether to perform evaluation operations only?")
        val_parser.add_argument(
            "--source",
            type=str,
            default="",
            help="Specifies the data source for inference. Can be an image path or directory. Supports a wide range of formats and sources, enabling flexible application across predict_parser types of input.",
        )
        val_parser.add_argument("--model_path", type=str, default="", help="The path where the trained model is located, used for prediction.")
        val_parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="Specifies the device for inference, a single GPU (device=cuda:0 or 0), multiple GPUs (device=0,1 or cuda,cuda:1). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.",
        )
        val_parser.add_argument(
            "--save_result",
            action="store_true",
            help="Allows you to save the model's inference results to a specified folder for use in the next evaluation. Only takes effect when inference evaluation is required.",
        )
        val_parser.add_argument(
            "--infer_workers",
            type=int,
            default=1,
            help="Number of worker processes for Model is inferring.",
        )
        val_parser.add_argument(
            "--eval_workers",
            type=int,
            default=-1,
            help="Number of worker processes for Model is evaluating. The default value is -1, which indicates the number of CPU cores.",
        )

        # *** 预测参数
        predict_parser.add_argument(
            "--source",
            type=str,
            required=True,
            help="Specifies the data source for inference. Can be an image path, video file or directory. Supports a wide range of formats and sources, enabling flexible application across predict_parser types of input.",
        )
        predict_parser.add_argument("--model_path", type=str, required=True, help="The path where the trained model is located, used for prediction.")
        predict_parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            help="Specifies the device for inference, a single GPU (device=cuda:0 or 0), multiple GPUs (device=0,1 or cuda,cuda:1). Allows users to select between CPU, a specific GPU, or other compute devices for model execution.",
        )
        predict_parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Number of worker processes for Model is predicting.",
        )
        predict_parser.add_argument(
            "--show", action="store_true", help="If True, displays the annotated images or videos in a window. Useful for immediate visual feedback during development or testing."
        )
        predict_parser.add_argument(
            "--save",
            action="store_true",
            help="Enables saving of the annotated images or videos to file. Useful for documentation, further analysis, or sharing results. Defaults to True when using CLI & False when used in Python.",
        )
        predict_parser.add_argument("--save_result", action="store_true", help="Allows you to save the model's inference results to a specified folder instead of saving visualization images.")

        # 额外新增参数
        self.add_train_args(train_parser)
        self.add_val_args(val_parser)
        self.add_predict_args(predict_parser)

    def parse(self):
        # *** 解析参数
        self.args = self.parser.parse_args()

        # *** 打印所有输入参数列表
        print("------------ Arguments -------------")
        for k, v in vars(self.args).items():
            print("%s=%s" % (str(k), str(v)), end=" ")
        print("\n------------------------------------")

        # *** 路径处理
        # 实验存储路径处理
        self.args.project = self.args.project or os.path.join(DEFAULT_CFG["dirs"]["runs_dir"], self.args.task)
        self.args.project = os.path.join(ROOT_DIR, self.args.project)
        self.args.name = self.args.name or self.args.mode
        # 设置实验的保存路径
        self.args.save_dir = os.path.join(self.args.project, self.args.name)
        # 如果 save_dir 已经存在，则自动增加一个数字后缀
        if os.path.exists(self.args.save_dir):
            # 是否重新恢复训练
            is_resume = self.args.mode == "train" and self.args.resume and (self.args.model_path == "" or self._is_subdirectory(self.args.save_dir, self.args.model_path))

            # 是否只验证模式
            is_only_eval = self.args.mode == "val" and self.args.only_eval

            # 是否不覆盖原有文件夹
            is_not_cover = not (is_resume or is_only_eval)

            if self.args.exist_ok:
                if is_not_cover:
                    shutil.rmtree(self.args.save_dir)
            elif is_not_cover:  # 除了少数模式(train: resume, val: only_eval, ...)，其他情况均重新创建序号文件夹
                for i in range(1, 100000):
                    if not os.path.exists(self.args.save_dir + f"_{i}"):
                        self.args.save_dir += f"_{i}"
                        break

        # *** 附加解析公开参数
        self.parse_public_args(self.args)

        # *** 不同模式下参数处理
        if self.args.mode == "train":
            # 设置实验中权重的保存路径
            self.args.save_weights_dir = os.path.join(self.args.save_dir, "weights")
            # 设置实验中日志的保存路径
            self.args.save_log_dir = os.path.join(self.args.save_dir, "log")
            # 创建目录
            os.makedirs(self.args.save_weights_dir, exist_ok=True)
            os.makedirs(self.args.save_log_dir, exist_ok=True)

            # 设置恢复训练路径
            if self.args.resume:
                # 如果恢复训练模型路径为空，则默认为项目路径下最后一个模型
                if self.args.model_path == "":
                    self.args.model_path = os.path.join(self.args.save_weights_dir, "model_last.pth")
                # 如果恢复训练模型不存在则抛出异常
                if not os.path.exists(self.args.model_path):
                    raise ValueError(f"Model path '{self.args.model_path}' is not exist.")

            # 附加解析训练参数
            self.parse_train_args(self.args)
        elif self.args.mode == "val":
            # 设置实验中结果的保存路径
            self.args.save_results_dir = os.path.join(self.args.save_dir, "results")

            # 是否仅进行评估
            if self.args.only_eval:
                # 如果只评估，需要保证预测结果路径有效，标签路径有效
                if not os.path.exists(self.args.save_results_dir):
                    raise ValueError(f"Predict results path '{self.args.save_results_dir}' is not exist. Unable to evaluate!")

                if not os.path.exists(self.args.label):
                    raise ValueError(f"Label path '{self.args.label}' is not exist. Unable to evaluate!")
            else:
                # 创建目录
                os.makedirs(self.args.save_results_dir, exist_ok=True)

            # 评估进程数目
            if self.args.eval_workers < 1:
                self.args.eval_workers = cpu_count()  # 如果异常则重置为cpu核心数目

            # 附加解析验证参数
            self.parse_val_args(self.args)
        elif self.args.mode == "predict":
            # 设置实验中结果图片和视频的保存路径
            self.args.save_shows_dir = os.path.join(self.args.save_dir, "shows")
            # 设置实验中结果的保存路径
            self.args.save_results_dir = os.path.join(self.args.save_dir, "results")
            # 创建目录
            os.makedirs(self.args.save_shows_dir, exist_ok=True)
            os.makedirs(self.args.save_results_dir, exist_ok=True)

            # 附加解析预测参数
            self.parse_predict_args(self.args)

        return self.args

    def add_public_args(self, public_parser):
        pass

    def add_train_args(self, train_parser):
        pass

    def add_val_args(self, val_parser):
        pass

    def add_predict_args(self, predict_parser):
        pass

    def parse_public_args(self, public_args):
        pass

    def parse_train_args(self, train_args):
        pass

    def parse_val_args(self, val_args):
        pass

    def parse_predict_args(self, predict_args):
        pass

    def _get_task_choices(self):
        task_choices = []

        task_dir = os.path.join(SRC_DIR, "engine")

        for task in os.listdir(task_dir):
            if os.path.isdir(os.path.join(task_dir, task)) and task != "base":
                task_choices.append(task)

        return task_choices

    def _is_subdirectory(parent, child):
        # 规范化路径
        parent = os.path.abspath(parent)
        child = os.path.abspath(child)

        # 检查子目录
        return os.path.commonprefix([parent, child]) == parent and parent != child
