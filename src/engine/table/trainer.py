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

        # Dataset reading class
        self.Dataset = get_dataset(args) if dataset is None else dataset

        # Models
        model = create_model(args.model)

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

        # Loss function
        loss = TableLoss((1, 1, 1, 1, 1)) if loss is None else loss
        loss_stats = loss.loss_stats

        # Verbose log output is not enabled
        if not self.args.verbose:
            self.__setattr__("log", super().log)

        # Parent class constructor
        super().__init__(model, optimizer, (loss_stats, loss))

        # Logs
        self.logger = Logger(args)

        # The initial metrics of the optimal model
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
        # Set the training initialization parameters
        # torch.autograd.set_detect_anomaly(True)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True  # Enable the baseline mode of CuDNN

        # Set the device and the total number of rounds of training
        self.set_device(args.device, args.master_batch, args.batch).set_total_epoch(args.epochs)

        # Load the dataset
        train_loader = DataLoader(self.Dataset(args.data, "train"), batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
        val_loader = DataLoader(self.Dataset(args.data, "val"), batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        start_epoch = 0

        if args.model_path != "":
            train_info = load_model(self.model, args.model_path, self.optimizer, args.resume, args.lr, args.lr_step)
            if isinstance(train_info, tuple):
                self.model, self.optimizer, start_epoch = train_info
            else:
                self.model = train_info

        # Model storage path
        last_model_path = os.path.join(args.save_weights_dir, "model_last.pth")
        best_model_path = os.path.join(args.save_weights_dir, "model_best.pth")

        for epoch in range(start_epoch + 1, args.epochs + 1):
            # Train and save training logs
            self._run_and_log("train", epoch, args.epochs, train_loader)

            # Whether the iterative model is saved
            is_save_period = False

            # Save the iterative model
            if args.save_period > 0 and epoch % args.save_period == 0:
                save_model(os.path.join(args.save_weights_dir, "model_{}.pth".format(epoch)), epoch, self.model, self.optimizer)
                is_save_period = True

            # Validation, if the verification effect is good, the optimal model will be saved
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
