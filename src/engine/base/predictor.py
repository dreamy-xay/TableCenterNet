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
        # Models
        self.model = model
        self.model.eval()

        # Other parameters
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
            - image (np.ndarray): Enter an image
            - args (tuple): Other arguments, which will be passed as arguments to the self.pre_process, self.process, self.post_process functions
            - kwargs (dict): Other arguments, which will be passed as arguments to the self.pre_process, self.process, self.post_process functions
        """

        # Image pre-processing to get model input
        input = self.pre_process(image, *args, **kwargs)

        # Transfer the input data to the specified device
        other_args = ()
        if isinstance(input, tuple):
            other_args = input[1:]
            input = input[0].to(self.device)
        else:
            input = input.to(self.device)

        # Make sure all GPU operations are complete
        if self.use_gpu:
            torch.cuda.synchronize()

        # The model infers and decodes to get the result
        output = self.process(input, *other_args, *args, **kwargs)

        # Make sure all GPU operations are complete
        if self.use_gpu:
            torch.cuda.synchronize()

        # If the subclass overrides the debug function, call the debug function for debugging
        if self.is_debug:
            self.debug(image, input, output)

        # Post-processing of model inference results (including operations such as decoding).
        result = self.post_process(*output if isinstance(output, tuple) else output, *args, **kwargs)

        # Make sure all GPU operations are complete
        if self.use_gpu:
            torch.cuda.synchronize()

        # Generate a result image based on the original image and the inference result
        result_image = self.generate(image, result)

        return result, result_image

    def pre_process(self, image, *args, **kwargs):
        """
        Model input preprocessing, which must be implemented

        Args:
            - image (np.ndarray): Enter an image
            - args (tuple): Other arguments, args arguments from function self.run_image
            - kwargs (dict): Other arguments, kwargs arguments from function self.run_image

        Returns:
            - input (torch. Tensor | Tuple[torch. Tensor, ...]): Model input or tuple of elements starting with model input
        """
        raise NotImplementedError("Subclasses must implement the 'pre_process' method.")

    def process(self, input, *args, **kwargs):
        """
        Model inference, which can include a decoding process, must be implemented

        Args:
            - input (torch. Tensor): Model input
            - args (tuple): Other arguments, from the returns[1:] parameter of function self.pre_process and the args argument of function self.run_image
            - kwargs (dict): Other arguments, kwargs arguments from function self.run_image

        Returns:
            - output (any): The output of model inference (which can include a decoding process).
        """
        raise NotImplementedError("Subclasses must implement the 'process' method.")

    def post_process(self, output, *args, **kwargs):
        """
        Post-processing of model inference results (including operations such as decoding) must be implemented

        Args:
            - output (torch. Tensor): Model inference results
            - args (tuple): Other arguments, from the returns[1:] parameter of the function self.process and the args argument of the function self.run_image
            - kwargs (dict): Other arguments, kwargs arguments from function self.run_image

        Returns:
            - result (any): The output of model inference (which can include a decoding process).
        """
        raise NotImplementedError("Subclasses must implement the 'post_process' method.")

    def debug(self, image, input, output):
        """
        Debugging of model inference results (including operations such as decoding).

        Args:
            - image (np.ndarray): Enter an image
            - input (torch. Tensor | Tuple[torch. Tensor, ...]): Model input or tuple of elements starting with model input
            - output (any): The output of model inference (which can include a decoding process).

        Returns: None
        """
        raise NotImplementedError("Subclasses must implement the 'debug' method.")

    def generate(self, image, result):
        """
        Generating a result image based on the original image and the inference result must be implemented

        Args:
            - image (np.ndarray): Enter an image
            - result (any): the result of model inference and post-processing

        Returns:
            - result_image (np.ndarray): Result image
        """
        raise NotImplementedError("Subclasses must implement the 'generate' method.")

    def run(self):
        """
        The predictor executes the main function of the process, which must be implemented

        Returns: None
        """
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    def predict(self, options):
        """
        Single-process, single-device prediction

        Args:
            - options: must contain the following attributes:
                - source: The path of the image or video folder
                - device: The name of the device, e.g. "cuda:0" or "cpu"
                - save: Whether to save the result image of image inference, which is generated by BasePredictor.generate
                - show: Whether to display the result image of image inference, which is generated by BasePredictor.generate
                - save_shows_dir: The path where the result image of the inference is saved

        Returns:
            - results: Returns a list of inference results, each of which is as follows
                - type: Result type, "image" or "video"
                - name: The name of the file for inference
                - result: The result of the inference, which is the return value of BasePredictor.post_process
        """
        # Set up the device
        self.set_device(options.device)

        # Start Forecasting
        media_processor = MediaProcessor(options.source, options.save_shows_dir, self.run_image)  # Picture and video loading runner
        results = media_processor.process(options.show, options.save)

        return results

    def parallel_predict(self, options):
        """
        Multi-process (any number of devices) parallel prediction

        Args:
            - options: must contain the following attributes:
                - source: The path of the image or video folder
                - workers: the number of processes per device
                - devices: A list of devices, such as ['cuda:0', 'cuda:1', 'cpu', ...], which can be obtained from BasePredictor._get_devices
                - save: Whether to save the result image of image inference, which is generated by BasePredictor.generate
                - show: Whether to display the result image of image inference, which is generated by BasePredictor.generate
                - save_shows_dir: The path where the result image of the inference is saved

        Returns:
            - results: Returns a list of inference results, each of which is as follows
                - type: Result type, "image" or "video"
                - name: The name of the file for inference
                - result: The result of the inference, which is the return value of BasePredictor.post_process
        """
        # Load all legend images
        file_path_list = MediaProcessor.get_file_paths(options.source)
        # Initialize the multi-process inference progress bar
        progress_bar = WorkerParallel.SharedProgressBar(total=len(file_path_list), desc="Parallel Processing Media", unit="file")

        # Set the function parameters for each process for a single inference
        predict_args_list = []
        for filename, file_path in file_path_list:
            predict_args_list.append((filename, file_path))

        # Set the sharing parameters for each process
        worker_args_list = []
        for _ in range(options.workers):
            for device in options.devices:
                cur_predictor = copy.copy(self)
                setattr(cur_predictor, "initialize_cuda_device", device)
                worker_args_list.append((progress_bar, options, cur_predictor))

        # Define a single-file prediction function
        def parallel_predict(bar, options, predictor, filename, file_path):
            initialize_cuda_device = getattr(predictor, "initialize_cuda_device", None)
            if initialize_cuda_device is not None:
                setattr(predictor, "initialize_cuda_device", None)
                predictor.set_device(initialize_cuda_device)
            media_processor = MediaProcessor(options.source, options.save_shows_dir, predictor.run_image)
            result = media_processor.process_file(bar, filename, file_path, options.show, options.save)
            return result

        # Start multi-process inference
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
