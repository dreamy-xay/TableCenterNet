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
import torch
from torch.utils.data import DataLoader
import os
import shutil
from engine.base.trainer import BaseTrainer
from .dataset import get_dataset
from .loss import TableLoss
from utils.logger import Logger
from utils.model import create_model, load_model, save_model


class TableTrainer(BaseTrainer):
    def __init__(self, args, loss=None, dataset=None):
        self.args = args

        # 数据集读取类
        self.Dataset = get_dataset(args) if dataset is None else dataset

        # 模型
        model = create_model(args.model)

        # 优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

        # 损失函数
        loss = TableLoss((1, 1, 1, 1, 1)) if loss is None else loss
        loss_stats = loss.loss_stats

        # 不启用详细日志输出
        if not self.args.verbose:
            self.__setattr__("log", super().log)

        # 父类构造函数
        super().__init__(model, optimizer, (loss_stats, loss))

        # 日志
        self.logger = Logger(args)

        # 最优模型的初始指标
        self.best = float("inf")

    def _run_and_log(self, mode, epoch, epochs, data_loader):
        if self.args.verbose:
            self.logger.write("---------------- [{} MODE epoch {}/{}] ----------------".format(mode.upper(), epoch, epochs), end_line=True)

        loss_dict = self.set_total_epoch(epochs).run_epoch(mode, epoch, data_loader)

        self.logger.write("epoch {} [{}/{}]: ".format(mode, epoch, epochs))

        for i, (k, v) in enumerate(loss_dict.items()):
            self.logger.scalar_summary("{}_{}".format(mode, k), v, epoch)
            self.logger.write("{}{}={:8f}".format("" if i == 0 else ", ", k, v))

        self.logger.write("", end_line=True)

        return loss_dict

    def log(self, output, loss_dict, iter_id, num_iters, bar):
        self.logger.write(self._parse_process_bar(bar)[1], end_line=True, prefix="timestamp:%H:%M:%S")

    def run(self):
        args = self.args
        # 训练初始化参数设置
        # torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True  # 启用 CuDNN 的基准模式

        # 设置设备和训练总轮数
        self.set_device(args.device, args.master_batch, args.batch).set_total_epoch(args.epochs)

        # 加载数据集
        train_loader = DataLoader(self.Dataset(args.data, "train"), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(self.Dataset(args.data, "val"), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        start_epoch = 0

        if args.model_path != "":
            train_info = load_model(self.model, args.model_path, self.optimizer, args.resume, args.lr, args.lr_step)
            if isinstance(train_info, tuple):
                self.model, self.optimizer, start_epoch = train_info
            else:
                self.model = train_info

        # 模型存储路径
        last_model_path = os.path.join(args.save_weights_dir, "model_last.pth")
        best_model_path = os.path.join(args.save_weights_dir, "model_best.pth")

        for epoch in range(start_epoch + 1, args.epochs + 1):
            # 训练并保存训练日志
            self._run_and_log("train", epoch, args.epochs, train_loader)

            # 是否保存了迭代模型
            is_save_period = False

            # 保存迭代模型
            if args.save_period > 0 and epoch % args.save_period == 0:
                save_model(os.path.join(args.save_weights_dir, "model_{}.pth".format(epoch)), epoch, self.model, self.optimizer)
                is_save_period = True

            # 进行验证，验证效果好将保存最优模型
            if args.val_epochs > 0 and epoch % args.val_epochs == 0:
                with torch.no_grad():
                    val_loss_dict = self._run_and_log("val", epoch // args.val_epochs, args.epochs // args.val_epochs, val_loader)

                if val_loss_dict["loss"] < self.best:
                    self.best = val_loss_dict["loss"]
                    save_model(best_model_path, epoch, self.model)

            if is_save_period:
                shutil.copyfile(os.path.join(args.save_weights_dir, "model_{}.pth".format(epoch)), last_model_path)
            else:
                save_model(last_model_path, epoch, self.model, self.optimizer)

            if epoch in args.lr_step:
                lr = args.lr * (0.1 ** (args.lr_step.index(epoch) + 1))
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

        self.logger.close()
