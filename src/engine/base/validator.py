#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 10:34:01
LastEditors: dreamy-xay
LastEditTime: 2024-11-16 19:25:15
"""
import copy
from .predictor import BasePredictor
from utils.media_processor import MediaProcessor
from utils.parallel import WorkerParallel


class BaseValidator(object):
    def __init__(self, predictor: BasePredictor = None):
        # 预测器
        self.predictor = predictor

    def set_device(self, device):
        self.predictor.set_device(device)
        return self

    def predict_one(self, image, *args, **kwargs):
        """
        Args:
            - image (np.ndarray): 输入图像
            - args (tuple): 其他参数，将作为参数传递给 self.pre_process, self.process, self.post_process 函数
            - kwargs (dict): 其他参数，将作为参数传递给 self.pre_process, self.process, self.post_process 函数

        Returns:
            - result (Any): 预测结果
        """
        if self.predictor is None:
            raise RuntimeError("This method cannot be called in evaluation mode.")

        self.predictor.__setattr__("generate", self.__generate)
        self.predictor.__setattr__("is_debug", False)

        return self.predictor.run_image(image, *args, **kwargs)

    def __generate(self, _, __):
        return None

    def run(self):
        """
        验证器执行流程的主函数，必须实现

        Returns: None
        """
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    def infer(self, options):
        """
        单进程单设备推理

        Args:
            - options: 必须含有以下属性
                - source: 图片或视频文件夹路径
                - device: 设备名称，如 "cuda:0" 或者 "cpu"

        Returns:
            - results: 返回推理的结果列表，列表的每一项具体如下
                - type: 结果类型，"image" 或者 "video"
                - name: 推理的文件名
                - result: 推理的结果，为 BasePredictor.post_process 的返回值
        """
        # 设置设备
        self.set_device(options.device)

        # 开始预测
        media_processor = MediaProcessor(options.source, None, self.predict_one, False)  # 图片视频加载运行器
        results = media_processor.process(False, False)

        return results

    def parallel_infer(self, options):
        """
        多进程（任意设备数）并行推理

        Args:
            - options: 必须含有以下属性
                - source: 图片或视频文件夹路径
                - infer_workers: 单设备进程数
                - devices: 设备列表，如 ['cuda:0', 'cuda:1', 'cpu', ...]，可以通过 BasePredictor._get_devices 获取

        Returns:
            - results: 返回推理的结果列表，列表的每一项具体如下
                - type: 结果类型，"image" 或者 "video"
                - name: 推理的文件名
                - result: 推理的结果，为 BasePredictor.post_process 的返回值
        """
        # 加载所有图例图片
        file_path_list = MediaProcessor.get_file_paths(options.source)
        # 初始化多进程推理进度条
        progress_bar = WorkerParallel.SharedProgressBar(total=len(file_path_list), desc="Parallel Processing Media", unit="file")

        # 设置每个进程单次推理的函数参数
        infer_args_list = []
        for filename, file_path in file_path_list:
            infer_args_list.append((filename, file_path))

        # 设置每个进程的共享参数
        worker_args_list = []
        for _ in range(options.infer_workers):
            for device in options.devices:
                cur_validator = copy.copy(self)
                setattr(cur_validator, "initialize_cuda_device", device)
                worker_args_list.append((progress_bar, options, cur_validator))

        # 定义一个单文件推理的函数
        def parallel_infer(bar, options, validator, filename, file_path):
            initialize_cuda_device = getattr(validator, "initialize_cuda_device", None)
            if initialize_cuda_device is not None:
                setattr(validator, "initialize_cuda_device", None)
                validator.set_device(initialize_cuda_device)
            media_processor = MediaProcessor(options.source, None, validator.predict_one, False)
            result = media_processor.process_file(bar, filename, file_path, False, False)
            return result

        # 开始多进程推理
        worker_parallel = WorkerParallel(parallel_infer, infer_args_list, worker_args_list)
        results = worker_parallel.run()
        progress_bar.close()

        return results
