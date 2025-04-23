#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 14:20:53
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 14:21:13
"""
import torch
import torch.distributed
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.autograd import Variable
from torch.nn.parallel._functions import Scatter
from multiprocessing import Pool, Queue, Manager, Process, cpu_count
from tqdm import tqdm


class _DataParallel(Module):
    r"""Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the batch
    dimension. In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is the
    same size (so that each GPU processes the same number of samples).

    See also: :ref:`cuda-nn-dataparallel-instead`

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel EXCEPT Tensors. All variables will be scattered on dim
    specified (default 0). Primitive types will be broadcasted, but all
    other types will be a shallow copy and can be corrupted if written to in
    the model's forward pass.

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)
        output_device: device location of output (default: device_ids[0])

    Example::

        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)
    """

    # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

    def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
        super(_DataParallel, self).__init__()

        if not torch.cuda.is_available():
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        if output_device is None:
            output_device = device_ids[0]
        self.dim = dim
        self.module = module
        self.device_ids = device_ids
        self.chunk_sizes = chunk_sizes
        self.output_device = output_device
        if len(self.device_ids) == 1:
            self.module.cuda(device_ids[0])

    def forward(self, *inputs, **kwargs):
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids, self.chunk_sizes)
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        replicas = self.replicate(self.module, self.device_ids[: len(inputs)])
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)

    def replicate(self, module, device_ids):
        return replicate(module, device_ids)

    def scatter(self, inputs, kwargs, device_ids, chunk_sizes):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=self.chunk_sizes)

    def parallel_apply(self, replicas, inputs, kwargs):
        return parallel_apply(replicas, inputs, kwargs, self.device_ids[: len(replicas)])

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)


def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
    r"""
    Slices variables into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not variables. Does not
    support Tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, Variable):
            return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
        assert not torch.is_tensor(obj), "Tensors not supported in scatter."
        if isinstance(obj, tuple):
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list):
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict):
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    return scatter_map(inputs)


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module: the module to evaluate in parallel
        inputs: inputs to the module
        device_ids: GPU ids on which to replicate module
        output_device: GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Variable containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[: len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)


def DataParallel(module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
    if chunk_sizes is None:
        return torch.nn.DataParallel(module, device_ids, output_device, dim)
    standard_size = True
    for i in range(1, len(chunk_sizes)):
        if chunk_sizes[i] != chunk_sizes[0]:
            standard_size = False
    if standard_size:
        return torch.nn.DataParallel(module, device_ids, output_device, dim)
    return _DataParallel(module, device_ids, output_device, dim, chunk_sizes)


class WorkerParallel:
    def __init__(self, func, fun_args_list, worker_args_list=[tuple() for _ in range(cpu_count())]):
        self.func = func
        self.fun_args_list = fun_args_list
        self.worker_args_list = worker_args_list

    @staticmethod
    def _worker(task_queue, result_list, func, worker_args):
        while not task_queue.empty():
            task_id, func_args = task_queue.get()
            result = func(*worker_args, *func_args)
            result_list.append((task_id, result))

    def run(self):
        # 创建任务队列并填充任务
        task_queue = Queue()
        for i, task in enumerate(self.fun_args_list):
            task_queue.put((i, task))  # 将任务 ID 和数据添加到队列中

        # 共享内存
        manager = Manager()

        # 创建结果队列，用于存放每个进程的推理结果
        result_list = manager.list()

        # 启动多个进程来处理任务
        processes = []
        for worker_args in self.worker_args_list:
            process = Process(target=self._worker, args=(task_queue, result_list, self.func, worker_args))
            process.start()
            processes.append(process)

        # 等待所有进程完成
        for process in processes:
            process.join()

        # 收集所有推理结果
        results = [None] * len(self.fun_args_list)
        for task_id, output in result_list:
            results[task_id] = output

        # 关闭共享内存
        manager.shutdown()

        return results

    class SharedProgressBar(tqdm):

        def __init__(self, *args, **kwargs):
            self.parent = super()
            self.parent.__init__(*args, **kwargs)
            self.manager = Manager()
            self.shared_progress = self.manager.Value("i", 0)

        def update(self, value):
            self.shared_progress.value += value
            self.parent.update(self.shared_progress.value - self.n)

        def close(self):
            self.manager.shutdown()
            self.parent.update(self.total - self.n)
            self.parent.close()


class ComputateParallel:
    def __init__(self, func, args_list, num_workers=cpu_count()):
        self.func = func
        self.args_list = args_list
        self.num_workers = num_workers
        self.tqdm_kwds = {}
        self.postfix_map = None

    def set_tqdm(self, postfix_map=None, **kwds):
        self.tqdm_kwds = kwds
        if "total" in kwds:
            del self.tqdm_kwds["total"]
        if callable(postfix_map):
            self.postfix_map = postfix_map
        return self

    def run(self, use_process_pool=True) -> list:
        if use_process_pool:
            # 设置全局函数包裹 self.func，防止 apply_async 函数序列化失败
            global _class_ComputateParallel_run_func_bind_

            # 定义函数
            def _class_ComputateParallel_run_func_bind_(*args):
                return self.func(*args)

            results = []
            # 使用进度条并同时创建进程池
            with Pool(processes=self.num_workers) as pool, tqdm(total=len(self.args_list), **self.tqdm_kwds) as bar:
                processes = []
                for args in self.args_list:
                    process = pool.apply_async(_class_ComputateParallel_run_func_bind_, args=args)
                    processes.append((args, process))

                for args, process in processes:
                    if self.postfix_map is not None:
                        bar.set_postfix(self.postfix_map(*args))
                    results.append(process.get())
                    bar.update(1)  # 每次更新进度条，表示一个任务已提交

                if self.postfix_map is not None:
                    bar.set_postfix_str()

            return results
        else:
            bar = WorkerParallel.SharedProgressBar(total=len(self.args_list), **self.tqdm_kwds)

            def func(bar, *args):
                if self.postfix_map is not None:
                    bar.set_postfix(self.postfix_map(*args))
                result = self.func(*args)
                bar.update(1)
                return result

            computate_parallel = WorkerParallel(func, self.args_list, [(bar,) for _ in range(self.num_workers)])

            results = computate_parallel.run()

            bar.close()

            return results


def run_func(target, timeout):
    # 创建一个队列用于传递返回值
    result_queue = Queue()

    # 定义内部函数执行目标函数并将结果返回
    def wrapper():
        try:
            result = target()
            result_queue.put(result)
        except:
            result_queue.put(None)

    # 创建一个进程来运行目标函数W
    process = Process(target=wrapper)

    # 启动进程
    process.start()

    # 等待进程结束，或超时
    process.join(timeout)

    if process.is_alive():  # 如果超时，终止进程
        process.terminate()
        process.join()  # 确保进程结束
        return None  # 超时返回 None
    else:
        # 获取进程的返回值
        if not result_queue.empty():
            return result_queue.get()
        return None  # 如果没有返回值，也返回 None
