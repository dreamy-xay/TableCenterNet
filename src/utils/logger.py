#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description: 
Version: 
Autor: dreamy-xay
Date: 2024-10-22 17:38:47
LastEditors: dreamy-xay
LastEditTime: 2024-10-22 17:38:52
"""
import os
import time
import sys
import torch


class Logger(object):
    def __init__(self, options, use_tensorboard=False):
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.use_tensorboard = True
        else:
            self.use_tensorboard = False

        self.save_options(options)

        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=options.save_log_dir)

        if not os.path.exists(options.save_log_dir):
            os.makedirs(options.save_log_dir)

        self.log = open(options.save_log_dir + "/log.txt", "a")
        self.start_line = True

    def write(self, txt, end_line=False, prefix="timestamp:%Y-%m-%d %H:%M"):
        log_txt = ""

        if self.start_line:
            if prefix.startswith("timestamp"):
                time_str = time.strftime(prefix[10:])
                log_txt += f"[{time_str}] {txt}"
            else:
                log_txt += f"{prefix} {txt}"
        else:
            log_txt += txt

        if end_line:
            log_txt += "\n"

        self.log.write(log_txt)

        if end_line:
            self.start_line = True
            self.log.flush()
        else:
            self.start_line = False

    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, step)

    @staticmethod
    def save_options(options):
        args = dict((name, getattr(options, name)) for name in dir(options) if not name.startswith("_"))
        file_name = os.path.join(options.save_dir, "args.txt")
        with open(file_name, "wt") as args_file:
            args_file.write("==> torch version: {}\n".format(torch.__version__))
            args_file.write("==> cudnn version: {}\n".format(torch.backends.cudnn.version()))
            args_file.write("==> Cmd:\n")
            args_file.write(str(sys.argv[1:]))
            args_file.write("\n==> Args:\n")
            for k, v in sorted(args.items()):
                args_file.write("  %s: %s\n" % (str(k), str(v)))
