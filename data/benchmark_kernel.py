import os

from data import common
from data import srdata
from data import kerneldata

import numpy as np

import torch
import torch.utils.data as data

class Benchmark_kernel(kerneldata.KERNELData):
    """
    Data generator for benchmark tasks
    """
    def __init__(self, args, name='', train=False):
        super(Benchmark_kernel, self).__init__(
            args, name=name, train=train
        )

        if args.process:
            if train:
                self.data_hr, self.data_lr = self._load(self.num_video)
            else:
                self.data_hr, self.data_lr, self.data_kernel = self._load(self.num_video)

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.args.data_root)
        self.dir_kernel = os.path.join(self.apath, self.args.test_kernel_path)
        self.dir_lr = os.path.join(self.apath, self.args.test_blur_path)
        self.dir_hr = os.path.join(self.apath, self.args.dir_hr)
        print("test Kernel path (Kernel):", self.dir_kernel)
        print("test video path (LR):", self.dir_lr)
        print("test video path (HR):", self.dir_hr)


    def _load_file(self, idx):
        lr, hr, kernel, filename = super(Benchmark_kernel, self)._load_file(idx=idx)
        return lr, hr, kernel, filename