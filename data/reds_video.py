import os

from data import common
from data import vsrdata

import numpy as np

import torch
import torch.utils.data as data

class REDS_VIDEO(vsrdata.VSRData):

    def __init__(self, args, name='REDS/train/', train=True):
        super(REDS_VIDEO, self).__init__(args, name=name, train=train)


    def _scan(self):
        names_hr, names_lr = super(REDS_VIDEO, self)._scan()
        print(len(names_hr), len(names_lr), print(self.begin), print(self.end))
        names_hr = names_hr[self.begin: self.end]
        names_lr = names_lr[self.begin: self.end]

        return names_hr, names_lr
        
    def _set_filesystem(self, dir_data):
        print("Loading REDS videos")
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'train_sharp')
        self.dir_lr = os.path.join(self.apath, 'train_sharp_bicubic/X4')
        print("Train video path (HR):", self.dir_hr)
        print("Train video path (LR):", self.dir_lr)
    
