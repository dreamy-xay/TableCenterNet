#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
Version:
Autor: dreamy-xay
Date: 2024-11-25 10:31:52
LastEditors: dreamy-xay
LastEditTime: 2025-04-26 18:13:50
"""
import torch
import torch.nn.functional as F


class DynamicWeightAveraging:
    """
    Dynamic Weight Averaging (DWA).

    Parameters:
        T (float, default=2.0): Softmax temperature.
    """

    def __init__(self, T=2.0):
        self.T = T  # Temperature parameters
        self.loss_buffer = [None, None]

    def __call__(self, losses):
        """
        Calculate the dynamic weight for a given loss.

        Parameters:
            losses (torch. Tensor): A tensor of the shape (task_num,) containing the loss for each task.

        Returns:
            torch. Tensor: A tensor of the shape (task_num,) that contains the calculated weights for each task.
        """
        # Update the loss buffer with the new loss value
        self.loss_buffer[0] = self.loss_buffer[1]
        self.loss_buffer[1] = losses

        # If there is no previous loss data, all tasks are weighted the same
        if self.loss_buffer[0] is None:
            weights = torch.ones_like(losses)
        else:
            # Calculate the weight of each task based on the last two losses
            w_i = self.loss_buffer[1] / (self.loss_buffer[0] + 1e-8)  # Add a small value to avoid dividing by zero
            weights = len(losses) * F.softmax(w_i / self.T, dim=-1)  # Apply softmax and scale

        return weights
