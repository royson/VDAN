import os
import glob
from importlib import import_module

from torch.utils.data import DataLoader

class Data:
    def __init__(self, args):
        self.args = args
        self.data_train = args.data_train
        self.data_test = args.data_test

        list_benchmarks = ['Set5', 'Set14', 'B100', 'Urban100']
        benchmark = self.data_test in list_benchmarks
        
        list_benchmarks_video = ['Vid4', 'val', 'REDS4']
        benchmark_video = self.data_test in list_benchmarks_video

        list_benchmarks_kernel = ['REDS4_Kernel', 'TKE']
        benchmark_kernel = self.data_test in list_benchmarks_kernel
        if not self.args.test_only:
            m_train = import_module('data.' + self.data_train.lower())
            trainset = getattr(m_train, self.data_train)(self.args)
            print(trainset, self.data_train)
            self.loader_train = DataLoader(
                trainset,
                batch_size=self.args.batch_size,
                shuffle=True,
                pin_memory=not self.args.cpu,
                drop_last=True
            )
        else:
            self.loader_train = None


        m_test = import_module('data.benchmark_kernel')
        testset = getattr(m_test, 'Benchmark_kernel')(self.args, name=args.data_test, train=False)

        self.loader_test = DataLoader(testset, batch_size=self.args.n_GPUs, shuffle=False, pin_memory=not self.args.cpu)

