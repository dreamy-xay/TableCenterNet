#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 10:32:40
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 14:07:32
"""
import torch
import os
from tqdm import tqdm
from utils.parallel import DataParallel
from utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch["input"])

        loss, loss_stats = self.loss(outputs, batch)

        return outputs[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(self, model, optimizer=None, loss=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_stats, self.loss = loss
        self.model_with_loss = ModleWithLoss(self.model, self.loss)
        self.total_epoch = None

        # 调试、日志等参数
        self.is_debug = not getattr(self, "debug").__qualname__.startswith("BaseTrainer.")
        self.is_log = not getattr(self, "log").__qualname__.startswith("BaseTrainer.")

    def reset_model_with_loss(self, model_with_loss_cls):
        self.model_with_loss = model_with_loss_cls(self.model, self.loss)

    def set_device(self, device, master_batch_size, batch_size):
        chunk_sizes, gpus = self._get_chunk(device, master_batch_size, batch_size)

        self.gpus = gpus
        if len(gpus) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            self.device = torch.device("cuda")

            print(f"Using {len(gpus)} GPUs: {gpus}, chunk_sizes: {chunk_sizes}.")
        else:
            self.device = torch.device("cpu")
            print("Using CPU.")

        if len(gpus) > 1:
            self.model_with_loss = DataParallel(self.model_with_loss, device_ids=list(range(len(gpus))), chunk_sizes=chunk_sizes).to(self.device)
        else:
            self.model_with_loss = self.model_with_loss.to(self.device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=self.device, non_blocking=True)

        return self

    def set_total_epoch(self, total_epoch):
        self.total_epoch = total_epoch

        return self

    def run_epoch(self, mode, epoch, data_loader, num_iters=-1):
        # 获取模型推理和损失计算模块
        model_with_loss = self.model_with_loss

        if mode == "train":
            model_with_loss.train()
        else:
            if len(self.gpus) > 1:
                model_with_loss = model_with_loss.module
            model_with_loss.eval()
            torch.cuda.empty_cache()

        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if num_iters <= 0 else num_iters
        bar = tqdm(total=num_iters, desc=f"{mode} epoch[{epoch}{'/' + str(self.total_epoch) if self.total_epoch is not None else ''}]")

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break

            for key in batch:
                batch[key] = batch[key].to(device=self.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()
            if mode == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loss_dict = {}
            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch["input"].size(0))
                loss_dict[l] = "{:.4f}".format(avg_loss_stats[l].avg)

            bar.set_postfix(loss_dict)
            bar.update(1)

            if self.is_debug:
                self.debug(batch, output, iter_id)

            if self.is_log:
                self.log(output, loss_dict, iter_id, num_iters, bar)

            del output, loss, loss_stats

        bar.close()

        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret["time"] = bar.format_dict["elapsed"] / 60.0

        return ret

    def val(self, epoch, data_loader, num_iters=-1):
        return self.run_epoch("val", epoch, data_loader, num_iters)

    def train(self, epoch, data_loader, num_iters=-1):
        return self.run_epoch("train", epoch, data_loader, num_iters)

    def debug(self, batch, output, iter_id):
        raise NotImplementedError("Subclasses must implement the 'debug' method.")

    def log(self, output, loss_dict, iter_id, num_iters, bar):
        raise NotImplementedError("Subclasses must implement the 'log' method.")

    def run(self):
        """
        训练器执行流程的主函数，必须实现

        Returns: None
        """
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    @staticmethod
    def _get_chunk(device, master_batch_size, batch_size):
        # cpu
        if device == "cpu":
            return [batch_size], []

        # gpus
        gpus = [int(gpu) for gpu in device.split(",")]
        gpus = list(filter(lambda x: x >= 0, gpus))

        num_gpus = len(gpus)

        # 多GPU训练时，每个GPU上的batch size
        if master_batch_size == -1:
            master_batch_size = batch_size // num_gpus
        rest_batch_size = batch_size - master_batch_size
        chunk_sizes = [master_batch_size]
        for i in range(num_gpus - 1):
            slave_chunk_size = rest_batch_size // (num_gpus - 1)
            if i < rest_batch_size % (num_gpus - 1):
                slave_chunk_size += 1
            chunk_sizes.append(slave_chunk_size)

        return chunk_sizes, gpus

    @staticmethod
    def _parse_process_bar(bar):
        bar_dict = bar.format_dict
        if "ncols" in bar_dict:
            bar_dict.pop("ncols")
        if "nrows" in bar_dict:
            bar_dict.pop("nrows")
        bar_output = bar.format_meter(**bar_dict)
        return bar_dict, bar_output
