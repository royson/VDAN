#system lib
import os
import time
import datetime
import random
import glob
from multiprocessing import Process
from multiprocessing import Queue

#pytorch required
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torchvision.utils import make_grid
from torch.autograd import Variable
#mathematics
import numpy as np
import math
import mmcv

#image lib
import imageio
import cv2
from PIL import Image
import PIL
from skimage.util import img_as_uint
#plot

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import collections
try:
    import accimage
except ImportError:
    accimage = None
#metrics funtion
#===============
def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calc_ssim(args,
                   img1,
                   img2,
                   input_order='CHW',
                   test_y_channel=True):
    """Calculate SSIM (structural similarity).
    Ref:
    Image quality assessment: From error visibility to structural similarity
    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.
    For three-channel images, SSIM is calculated for each channel and then
    averaged.
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: ssim result.
    """
    img1 = img1[0].data.mul(255.0 / args.rgb_range).cpu().numpy()
    img2 = img2[0].data.mul(255.0 / args.rgb_range).cpu().numpy()
    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    crop_border = args.scale
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if args.converty:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def calc_psnr(args,
                   img1,
                   img2,
                   input_order='CHW',
                   test_y_channel=True):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """

    img1 = img1[0].data.mul(255.0 / args.rgb_range).cpu().numpy()
    img2 = img2[0].data.mul(255.0 / args.rgb_range).cpu().numpy()

    assert img1.shape == img2.shape, (
        f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    crop_border = args.scale
    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if args.converty:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    
    return torch.tensor(20. * np.log10(255. / np.sqrt(mse)))

def to_y_channel(img):
    """Change to Y channel of YCbCr.
    Args:
        img (ndarray): Images with range [0, 255].
    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = mmcv.rgb2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255

def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose((1, 2, 0))
    return img


# post process of image

def postprocess(*images, rgb_range, ycbcr_flag, device):
    def _postprocess(img, rgb_coefficient, ycbcr_flag, device):
        if ycbcr_flag:
            mean_YCbCr = torch.Tensor([109]).to(device)
            out = (img.mul(rgb_coefficient) + mean_YCbCr).clamp(16, 235).div(rgb_coefficient)
        elif img.shape[2] == 3:
            #mean_RGB = torch.Tensor([123.68, 116.779, 103.939]).to(device)
            #mean_RGB = mean_RGB.reshape([1, 3, 1, 1])
            out = (img.mul(rgb_coefficient)).clamp(0, 255).round().div(rgb_coefficient)
        else:
            mean_YCbCr = torch.Tensor([109]).to(device)
            out = (img.mul(rgb_coefficient) + mean_YCbCr).clamp(0, 255).round()
            out.div_(rgb_coefficient)

        return out

    rgb_coefficient = 255 / rgb_range
    return [_postprocess(img, rgb_coefficient, ycbcr_flag, device) for img in images]

"""SRMD-preprocess on image 
ref: https://github.com/yuanjunchai/IKC/blob/master/codes/utils/util.py
ref: IKC: Blind Super-Resolution With Iterative Kernel Correction : https://arxiv.org/abs/1904.03377
"""
# PCA of Kernel
def PCA(data, k=2):
    X = torch.from_numpy(data)
    X_mean = torch.mean(X, 0)
    X = X - X_mean.expand_as(X)
    U, S, V = torch.svd(torch.t(X))
    return U[:, :k] # PCA matrix

def cal_sigma(sig_x, sig_y, radians):
    D = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    U = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), 1 * np.cos(radians)]])
    sigma = np.dot(U, np.dot(D, U.T))
    return sigma


def anisotropic_gaussian_kernel(l, sigma_matrix, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((l * l, 1)), yy.reshape(l * l, 1))).reshape(l, l, 2)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(xy, inverse_sigma) * xy, 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def isotropic_gaussian_kernel(l, sigma, tensor=False):
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return torch.FloatTensor(kernel / np.sum(kernel)) if tensor else kernel / np.sum(kernel)


def random_anisotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, scaling=3, l=21, tensor=False):
    pi = np.random.random() * math.pi * 2 - math.pi
    x = np.random.random() * (sig_max - sig_min) + sig_min
    y = np.clip(np.random.random() * scaling * x, sig_min, sig_max)
    sig = cal_sigma(x, y, pi)
    k = anisotropic_gaussian_kernel(l, sig, tensor=tensor)
    return k


def random_isotropic_gaussian_kernel(sig_min=0.2, sig_max=4.0, l=21, tensor=False):
    x = np.random.random() * (sig_max - sig_min) + sig_min
    k = isotropic_gaussian_kernel(l, x, tensor=tensor)
    return k

def random_real_world_kernel(seq_id, i):
    data_root = '../../dataset/REDS4/sth_kernel'
    seq_path = os.path.join(data_root, str(seq_id))
    seq_kernel_name = sorted(glob.glob(os.path.join(seq_path, '*.npy')))
    seq_length = len(seq_kernel_name)

    return k
def random_gaussian_kernel(l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=False):
    if np.random.random() < rate_iso:
        return random_isotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, tensor=tensor)
    else:
        return random_anisotropic_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, scaling=scaling, tensor=tensor)

def load_kernel_seq(data_root, batch, n_seq):
    kernel_seq_samples = []
    sample_seq_id = np.random.randint(1, 80, size=(batch//n_seq))
    for i in sample_seq_id:
        seq_path = os.path.join(data_root, str(i))
        seq_k_names = sorted(glob.glob(os.path.join(seq_path, '*.npy')))
        seq_len = len(seq_k_names)
        if n_seq >= seq_len:
            print('not enough kernels for load:', seq_path)
        high = np.random.randint(n_seq, seq_len)
        select_ks = seq_k_names[high-n_seq:high]
        for k in select_ks:
            kernel_seq_samples.append(np.load(k))
    
    return kernel_seq_samples


def random_batch_kernel(batch, l=21, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, tensor=True, n_seq=3, real=0.1, real_path=''):
    batch_kernel = np.zeros((batch, l, l))
    if np.random.random() < real:
        data_root = real_path
        real_kernel = load_kernel_seq(data_root, batch, n_seq)
        for i in range(batch):
            batch_kernel[i] = real_kernel[i]
        return torch.FloatTensor(batch_kernel) if tensor else batch_kernel
    for i in range(batch):
        batch_kernel[i] = random_gaussian_kernel(l=l, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, tensor=False)
    return torch.FloatTensor(batch_kernel) if tensor else batch_kernel

def b_GPUVar_Bicubic(variable, scale):
    tensor = variable.cpu().data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_view = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_view[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_view = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_view


def b_CPUVar_Bicubic(variable, scale):
    tensor = variable.data
    B, C, H, W = tensor.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = tensor.view((B*C, 1, H, W))
    re_tensor = torch.zeros((B*C, 1, H_new, W_new))
    for i in range(B*C):
        img = to_pil_image(tensor_v[i])
        re_tensor[i] = to_tensor(resize(img, (H_new, W_new), interpolation=Image.BICUBIC))
    re_tensor_v = re_tensor.view((B, C, H_new, W_new))
    return re_tensor_v


def random_batch_noise(batch, high, rate_cln=1.0):
    noise_level = np.random.uniform(size=(batch, 1)) * high
    noise_mask = np.random.uniform(size=(batch, 1))
    noise_mask[noise_mask < rate_cln] = 0
    noise_mask[noise_mask >= rate_cln] = 1
    return noise_level * noise_mask


def b_GaussianNoising(tensor, sigma, mean=0.0, noise_size=None, min=0.0, max=1.0):
    if noise_size is None:
        size = tensor.size()
    else:
        size = noise_size
    noise = torch.mul(torch.FloatTensor(np.random.normal(loc=mean, scale=1.0, size=size)), sigma.view(sigma.size() + (1, 1)))
    return torch.clamp(noise + tensor, min=min, max=max)

class BatchSRKernel(object):
    def __init__(self, l=21, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, n_seq=3, real=0.1, real_path=''):
        self.l = l
        self.sig = sig
        self.sig_min = sig_min
        self.sig_max = sig_max
        self.rate = rate_iso
        self.scaling = scaling
        self.n_seq = n_seq
        self.real = real
        self.real_path = real_path
    def __call__(self, random, batch, tensor=False):
        if random == True: #random kernel
            return random_batch_kernel(batch, l=self.l, sig_min=self.sig_min, sig_max=self.sig_max, rate_iso=self.rate,
                                       scaling=self.scaling, tensor=tensor, n_seq=self.n_seq, real=self.real, real_path=self.real_path)


class PCAEncoder(object):
    def __init__(self, weight, cuda=False):
        self.weight = weight #[l^2, k]
        self.size = self.weight.size()
        if cuda:
            self.weight = Variable(self.weight).cuda()
        else:
            self.weight = Variable(self.weight)

    def __call__(self, batch_kernel):
        B, H, W = batch_kernel.size() #[B, l, l]
        return torch.bmm(batch_kernel.view((B, 1, H * W)), self.weight.expand((B, ) + self.size)).view((B, -1))


class BatchBlur(nn.Module):
    def __init__(self, l=15):
        super(BatchBlur, self).__init__()
        self.l = l
        if l % 2 == 1:
            self.pad = nn.ReflectionPad2d(l // 2)
        else:
            self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        # self.pad = nn.ZeroPad2d(l // 2)

    def forward(self, input, kernel):
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]

        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            input_CBHW = pad.view((1, C * B, H_p, W_p))
            kernel_var = kernel.contiguous().view((B, 1, self.l, self.l)).repeat(1, C, 1, 1).view((B * C, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, groups=B*C).view((B, C, H, W))


class SRMDPreprocessing(object):
    def __init__(self, scale, pca, random, para_input=10, kernel=21, noise=True, cuda=False, sig=2.6, sig_min=0.2, sig_max=4.0, rate_iso=1.0, scaling=3, rate_cln=0.2, noise_high=0.08, n_seq=3, real=0.1, real_path=''):
        self.encoder = PCAEncoder(pca, cuda=cuda)
        self.kernel_gen = BatchSRKernel(l=kernel, sig=sig, sig_min=sig_min, sig_max=sig_max, rate_iso=rate_iso, scaling=scaling, n_seq=n_seq, real=real, real_path=real_path)
        self.blur = BatchBlur(l=kernel)
        self.para_in = para_input
        self.l = kernel
        self.noise = noise
        self.scale = scale
        self.cuda = cuda
        self.rate_cln = rate_cln
        self.noise_high = noise_high
        self.random = random

    def __call__(self, hr_tensor, kernel=False):
        ### hr_tensor is tensor, not cuda tensor
        B, C, H, W = hr_tensor.size()
        b_kernels = Variable(self.kernel_gen(self.random, B, tensor=True)).cuda() if self.cuda else Variable(self.kernel_gen(self.random, B, tensor=True))
        # blur
        if self.cuda:
            hr_blured_var = self.blur(Variable(hr_tensor).cuda(), b_kernels)
        else:
            hr_blured_var = self.blur(Variable(hr_tensor), b_kernels)
        # kernel encode
        kernel_code = self.encoder(b_kernels) # B x self.para_input
        # Down sample
        if self.cuda:
            lr_blured_t = b_GPUVar_Bicubic(hr_blured_var, self.scale)
        else:
            lr_blured_t = b_CPUVar_Bicubic(hr_blured_var, self.scale)

        # Noisy
        if self.noise:
            Noise_level = torch.FloatTensor(random_batch_noise(B, self.noise_high, self.rate_cln))
            lr_noised_t = b_GaussianNoising(lr_blured_t, Noise_level)
        else:
            Noise_level = torch.zeros((B, 1))
            lr_noised_t = lr_blured_t

        if self.cuda:
            Noise_level = Variable(Noise_level).cuda()
            re_code = kernel_code
            lr_re = Variable(lr_noised_t).cuda()
        else:
            Noise_level = Variable(Noise_level)
            re_code = kernel_code
            lr_re = Variable(lr_noised_t)
        return (lr_re, re_code, b_kernels) if kernel else (lr_re, re_code)


# image type function
def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.
    See :class:`~torchvision.transforms.ToPIlImage` for more details.
    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes
    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def resize(img, size, interpolation=Image.BILINEAR):
    """Resize the input PIL Image to the given size.
    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)