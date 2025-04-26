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
        # Predictor
        self.predictor = predictor

    def set_device(self, device):
        self.predictor.set_device(device)
        return self

    def predict_one(self, image, *args, **kwargs):
        """
        Args:
            - image (np.ndarray): Enter an image
            - args (tuple): Other arguments, which will be passed as arguments to the self.pre_process, self.process, self.post_process functions
            - kwargs (dict): Other arguments, which will be passed as arguments to the self.pre_process, self.process, self.post_process functions

        Returns:
            - result (Any): predicts the outcome
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
        The main function of the validator execution process, which must be implemented

        Returns: None
        """
        raise NotImplementedError("Subclasses must implement the 'run' method.")

    def infer(self, options):
        """
        Single-process, single-device inference

        Args:
            - options: must contain the following attributes:
                - source: The path of the image or video folder
                - device: The name of the device, e.g. "cuda:0" or "cpu"

        Returns:
            - results: Returns a list of inference results, each of which is as follows
                - type: Result type, "image" or "video"
                - name: The name of the file for inference
                - result: The result of the inference, which is the return value of BasePredictor.post_process
        """
        # Set up the device
        self.set_device(options.device)

        # Start Forecasting
        media_processor = MediaProcessor(options.source, None, self.predict_one, False)  # Picture and video loading runner
        results = media_processor.process(False, False)

        return results

    def parallel_infer(self, options):
        """
        Multi-process (any number of devices) parallel inference

        Args:
            - options: must contain the following attributes:
                - source: The path of the image or video folder
                - infer_workers: the number of processes per device
                - devices: A list of devices, such as ['cuda:0', 'cuda:1', 'cpu', ...], which can be obtained from BasePredictor._get_devices

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
        infer_args_list = []
        for filename, file_path in file_path_list:
            infer_args_list.append((filename, file_path))

        # Set the sharing parameters for each process
        worker_args_list = []
        for _ in range(options.infer_workers):
            for device in options.devices:
                cur_validator = copy.copy(self)
                setattr(cur_validator, "initialize_cuda_device", device)
                worker_args_list.append((progress_bar, options, cur_validator))

        # Define a single-file inference function
        def parallel_infer(bar, options, validator, filename, file_path):
            initialize_cuda_device = getattr(validator, "initialize_cuda_device", None)
            if initialize_cuda_device is not None:
                setattr(validator, "initialize_cuda_device", None)
                validator.set_device(initialize_cuda_device)
            media_processor = MediaProcessor(options.source, None, validator.predict_one, False)
            result = media_processor.process_file(bar, filename, file_path, False, False)
            return result

        # Start multi-process inference
        worker_parallel = WorkerParallel(parallel_infer, infer_args_list, worker_args_list)
        results = worker_parallel.run()
        progress_bar.close()

        return results
