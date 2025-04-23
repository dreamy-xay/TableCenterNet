#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 10:34:01
LastEditors: dreamy-xay
LastEditTime: 2024-10-28 15:13:57
"""
import torch
import os
import copy
from utils.media_processor import MediaProcessor
from utils.parallel import WorkerParallel


class BasePredictor(object):
    def __init__(self, model):
        # 模型
        self.model = model
        self.model.eval()

        # 其他参数
        self.is_debug = not getattr(self, "debug").__qualname__.startswith("BasePredictor.")

    def set_device(self, device):
        gpu_str = self._get_gpu_str(device)

        self.use_gpu = gpu_str is not None

        if self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

            self.device = torch.device("cuda")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        return self

    def run_image(self, image, *args, **kwargs):
        """
        Args:
            - image (np.ndarray): 输入图像
            - args (tuple): 其他参数，将作为参数传递给 self.pre_process, self.process, self.post_process 函数
            - kwargs (dict): 其他参数，将作为参数传递给 self.pre_process, self.process, self.post_process 函数
        """

        # 图像预处理，的得到模型输入
        input = self.pre_process(image, *args, **kwargs)

        # 将输入数据转移到指定设备上
        other_args = ()
        if isinstance(input, tuple):
            other_args = input[1:]
            input = input[0].to(self.device)
        else:
            input = input.to(self.device)

        # 确保所有 GPU 操作完成
        if self.use_gpu:
            torch.cuda.synchronize()

        # 模型推理并解码得到结果
        output = self.process(input, *other_args, *args, **kwargs)

        # 确保所有 GPU 操作完成
        if self.use_gpu:
            torch.cuda.synchronize()

        # 如果子类覆写了 debug 函数，则调用 debug 函数进行调试
        if self.is_debug:
            self.debug(image, input, output)

        # 模型推理（含解码等操作）结果后处理
        result = self.post_process(*output if isinstance(output, tuple) else output, *args, **kwargs)

        # 确保所有 GPU 操作完成
        if self.use_gpu:
            torch.cuda.synchronize()

        # 根据原图像和推理结果生成结果图像
        result_image = self.generate(image, result)

        return result, result_image

    def pre_process(self, image, *args, **kwargs):
        """
        模型输入预处理，必须实现

        Args:
            - image (np.ndarray): 输入图像
            - args (tuple): 其他参数，来自函数 self.run_image 的 args 参数
            - kwargs (dict): 其他参数，来自函数 self.run_image 的 kwargs 参数

        Returns:
            - input (torch.Tensor | Tuple[torch.Tensor, ...]): 模型输入或者以模型输入为首元素的元组
        """
        raise NotImplementedError("Subclasses must implement the 'pre_process' method.")

    def process(self, input, *args, **kwargs):
        """
        模型推理（可以含解码过程），必须实现

        Args:
            - input (torch.Tensor): 模型输入
            - args (tuple): 其他参数，来自函数 self.pre_process 的 returns[1:] 参数和函数 self.run_image 的 args 参数
            - kwargs (dict): 其他参数，来自函数 self.run_image 的 kwargs 参数

        Returns:
            - output (Any): 模型推理（可以含解码过程）输出
        """
        raise NotImplementedError("Subclasses must implement the 'process' method.")

    def post_process(self, output, *args, **kwargs):
        """
        模型推理（含解码等操作）结果后处理，必须实现

        Args:
            - output (torch.Tensor): 模型推理结果
            - args (tuple): 其他参数，来自函数 self.process 的 returns[1:] 参数和函数 self.run_image 的 args 参数
            - kwargs (dict): 其他参数，来自函数 self.run_image 的 kwargs 参数

        Returns:
            - result (Any): 模型推理（可以含解码过程）输出
        """
        raise NotImplementedError("Subclasses must implement the 'post_process' method.")

    def debug(self, image, input, output):
        """
        模型推理（含解码等操作）结果调试

        Args:
            - image (np.ndarray): 输入图像
            - input (torch.Tensor | Tuple[torch.Tensor, ...]): 模型输入或者以模型输入为首元素的元组
            - output (Any): 模型推理（可以含解码过程）输出

        Returns: None
        """
        raise NotImplementedError("Subclasses must implement the 'debug' method.")

    def generate(self, image, result):
        """
        根据原图像和推理结果生成结果图像，必须实现

        Args:
            - image (np.ndarray): 输入图像
            - result (Any): 模型推理并后处理的结果

        Returns:
            - result_image (np.ndarray): 结果图像
        """
        raise NotImplementedError("Subclasses must implement the 'generate' method.")

    def run(self):
        """
        预测器执行流程的主函数，必须实现

        Returns: None
        """
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    def predict(self, options):
        """
        单进程单设备预测

        Args:
            - options: 必须含有以下属性
                - source: 图片或视频文件夹路径
                - device: 设备名称，如 "cuda:0" 或者 "cpu"
                - save: 是否保存图片推理的结果图像，由 BasePredictor.generate 生成
                - show: 是否展示图片推理的结果图像，由 BasePredictor.generate 生成
                - save_shows_dir: 推理的结果图像保存路径

        Returns:
            - results: 返回推理的结果列表，列表的每一项具体如下
                - type: 结果类型，"image" 或者 "video"
                - name: 推理的文件名
                - result: 推理的结果，为 BasePredictor.post_process 的返回值
        """
        # 设置设备
        self.set_device(options.device)

        # 开始预测
        media_processor = MediaProcessor(options.source, options.save_shows_dir, self.run_image)  # 图片视频加载运行器
        results = media_processor.process(options.show, options.save)

        return results

    def parallel_predict(self, options):
        """
        多进程（任意设备数）并行预测

        Args:
            - options: 必须含有以下属性
                - source: 图片或视频文件夹路径
                - workers: 单设备进程数
                - devices: 设备列表，如 ['cuda:0', 'cuda:1', 'cpu', ...]，可以通过 BasePredictor._get_devices 获取
                - save: 是否保存图片推理的结果图像，由 BasePredictor.generate 生成
                - show: 是否展示图片推理的结果图像，由 BasePredictor.generate 生成
                - save_shows_dir: 推理的结果图像保存路径

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
        predict_args_list = []
        for filename, file_path in file_path_list:
            predict_args_list.append((filename, file_path))

        # 设置每个进程的共享参数
        worker_args_list = []
        for _ in range(options.workers):
            for device in options.devices:
                cur_predictor = copy.copy(self)
                setattr(cur_predictor, "initialize_cuda_device", device)
                worker_args_list.append((progress_bar, options, cur_predictor))

        # 定义一个单文件预测的函数
        def parallel_predict(bar, options, predictor, filename, file_path):
            initialize_cuda_device = getattr(predictor, "initialize_cuda_device", None)
            if initialize_cuda_device is not None:
                setattr(predictor, "initialize_cuda_device", None)
                predictor.set_device(initialize_cuda_device)
            media_processor = MediaProcessor(options.source, options.save_shows_dir, predictor.run_image)
            result = media_processor.process_file(bar, filename, file_path, options.show, options.save)
            return result

        # 开始多进程推理
        worker_parallel = WorkerParallel(parallel_predict, predict_args_list, worker_args_list)
        results = worker_parallel.run()
        progress_bar.close()

        return results

    @staticmethod
    def _get_gpu_str(device):
        if device == "cuda":
            return "0"
        elif device.startswith("cuda"):
            device_info = device.split(":")
            if len(device_info) == 2 and device_info[0] == "cuda" and device_info[1].isdigit():
                return device_info[1]
        elif device.isdigit():
            return device

        return None

    @staticmethod
    def _get_devices(device_str):
        devices = BasePredictor._get_gpu_str(device_str)
        if devices is None:
            if "," in device_str:
                devices = []
                for cur_device_str in device_str.split(","):
                    devices.extend(BasePredictor._get_devices(cur_device_str))
            else:
                devices = [device_str]
        else:
            devices = [f"cuda:{devices}"]

        return devices
