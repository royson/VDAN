import random

import numpy as np
import skimage.io as sio
import skimage.color as sc
import skimage.transform as st

import torch
from torchvision import transforms

import os 
import cv2 
"""
Repository for common functions required for manipulating data
"""


def get_patch(*args, patch_size=17, scale=1, ix=1, iy=1):
    """
    Get patch from an image
    """
    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret

def get_degration_patch(*args, patch_size=17, scale=1, ix=1, iy=1, kernel=None):
    """
    Get patch from an image
    """
    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    tx, ty = scale * ix, scale * iy

    hr = args[1][ty:ty + tp, tx:tx + tp, :]
    lr = sisr.srmd_degradation(hr, kernel, scale)
    lr = np.clip(lr, 0., 255.).round().astype('uint8')
    ret = [ 
        lr,
        hr
    ]

    return ret

def get_frame_patch(*args, patch_size=17, scale=1, ix=1, iy=1):

    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        args[1][iy:iy + ip, ix:ix + ip, :]
    ]

    return ret

def get_kernel_patch(*args, patch_size=17, scale=1, ix=1, iy=1):

    ih, iw, _ = args[0].shape

    ip = patch_size
    tp = scale * ip

    tx, ty = scale * ix, scale * iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        args[1][iy:iy + ip, ix:ix + ip, :]
    ]

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        #print(img.ndim)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = sc.rgb2ycbcr(img)
            #print('num:', np.max(img))
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]


def np2Tensor(*args, rgb_range=255, n_colors=1):
    def _np2Tensor(img):
        # NHWC -> NCHW
        if img.shape[2] == 3 and n_colors == 3:
            img = img 
        elif img.shape[2] == 3 and n_colors == 1:
            mean_YCbCr = np.array([109, 0, 0])
            img = img - mean_YCbCr
        else:
            mean_YCbCr = np.array([109])
            img = img - mean_YCbCr

        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        #print('tensor:',torch.max(tensor))
        tensor.mul_(rgb_range / 255.)
        #print('tensor/255',torch.min(tensor), torch.max(tensor))

        return tensor

    return [_np2Tensor(a) for a in args]

def kernel2Tensor(*args, rgb_range=255, n_colors=1, norm=True):
    def _kernel2Tensor(img, norm):
        # NHWC -> NCHW
        if norm == True: 
            img = img.astype('float64') / np.max(img)

        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()

        return tensor

    return [_kernel2Tensor(a, norm) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = np.rot90(img)
        
        return img

    return [_augment(a) for a in args]