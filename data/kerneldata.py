import sys
import os
import glob
import time
import skimage.color as sc
from data import common
import pickle
import numpy as np
import imageio
import random
import torch
import torch.utils.data as data
import cv2
from scipy.io import loadmat

class KERNELData(data.Dataset):
    def __init__(self, args, name='', train=True):
        self.args = args
        self.name = name
        self.train = train
        self.scale = args.scale
        self.idx_scale = 0
        self.n_seq = args.n_sequence
        print("n_seq:", args.n_sequence)
        print("n_frames_per_video:", args.n_frames_per_video)
        # self.image_range : need to make it flexible in the test area
        self.img_range = 100
        self.n_frames_video = []
        data_range = [r.split('-') for r in args.data_range.split('/')]
        print(data_range)
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]
        print(list(map(lambda x: int(x), data_range)))
        self.begin, self.end = list(map(lambda x: int(x), data_range))
        print(self.begin, self.end)


        self._set_filesystem(args.dir_data_test)

        self.images_hr, self.images_lr, self.images_kernels = self._scan()
        self.num_video = len(self.images_lr)
        print("Number of videos to load:", self.num_video)
        if train:
            self.repeat = args.test_every // max((self.num_video // self.args.batch_size), 1)

    def _scan(self):
        """
        Returns a list of image directories
        """

        if self.args.real:
            vid_lr_names = sorted(glob.glob(os.path.join(self.dir_lr, '*')))
            names_lr = []
            for vid_lr_name in vid_lr_names:
                lr_dir_names = sorted(glob.glob(os.path.join(vid_lr_name, self.args.post_fix)))
                names_lr.append(lr_dir_names)
                self.n_frames_video.append(len(lr_dir_names))
            return None, names_lr, None
        else:
            vid_hr_names = sorted(glob.glob(os.path.join(self.dir_hr, '*')))
            vid_lr_names = sorted(glob.glob(os.path.join(self.dir_lr, '*')))
            vid_kernel_names = sorted(glob.glob(os.path.join(self.dir_kernel, '*')))

            print(len(vid_hr_names), len(vid_lr_names))
            assert len(vid_hr_names) == len(vid_lr_names)

            names_hr = []
            names_lr = []
            names_kernel = []

            for vid_hr_name, vid_lr_name, vid_kernel_name in zip(vid_hr_names, vid_lr_names, vid_kernel_names):
                hr_dir_names = sorted(glob.glob(os.path.join(vid_hr_name, self.args.post_fix)))
                lr_dir_names = sorted(glob.glob(os.path.join(vid_lr_name, self.args.post_fix)))
                vid_kernel_names = sorted(glob.glob(os.path.join(vid_kernel_name, '*.npy')))
                names_hr.append(hr_dir_names)
                names_lr.append(lr_dir_names)
                names_kernel.append(vid_kernel_names)
                self.n_frames_video.append(len(hr_dir_names))
            return names_hr, names_lr, names_kernel

    def _load(self, n_videos):

        if self.args.real:
            data_lr = []
            for idx in range(n_videos):
                if idx % 10 == 0:
                    print("Loading vide %d" %idx)
                lrs, _, _, _ = self._load_file(idx)
                lrs = np.array([imageio.imread(lr_name) for lr_name in self.images_lr[idx]])
                data_lr.append(lrs)
            return None, data_lr, None
        else:
            data_lr = []
            data_hr = []
            data_kernel = []
            for idx in range(n_videos):
                if idx % 10 == 0:
                    print("Loading vide %d" %idx)
                lrs, hrs, kernels, _ = self._load_file(idx)
                hrs = np.array([imageio.imread(hr_name) for hr_name in self.images_hr[idx]])
                lrs = np.array([imageio.imread(lr_name) for lr_name in self.images_lr[idx]])
                if self.args.fix_kernel:
                    fix_kernel = self.images_kernels[idx][0]
                    kernels = np.array([np.load(fix_kernel) for kernel_name in self.images_kernels[idx]])
                else: 
                    kernels = np.array([np.load(kernel_name) for kernel_name in self.images_kernels[idx]])
                data_lr.append(lrs)
                data_hr.append(hrs)
                data_kernel.append(kernels)
            return data_hr, data_lr, data_kernel

    def _set_filesystem(self, dir_data):
        print(dir_data)
        
    def __getitem__(self, idx):
        if self.args.real:
            if self.args.process:
                lrs, filenames = self._load_file_from_loaded_data(idx)
            else:
                lrs, filenames = self._load_file(idx)
            
            if len(lrs.shape) == 4:
                b, ih, iw, _ = lrs.shape
            else:
                lrs = np.array([np.expand_dims(lr, axis=2) for lr in lrs])
                b, ih, iw, _ = lrs.shape
                #print(lrs.shape)

            ip = self.args.patch_size

            ix = random.randrange(0, iw - ip + 1)
            iy = random.randrange(0, ih - ip + 1)

            patches = [self.get_patch(lr, None, ix, iy) for lr, hr in zip(lrs, None)]
            lrs = np.array([patch[0] for patch in patches])
            lrs = np.array(common.set_channel(*lrs, n_channels=self.args.n_colors))
            lr_tensors = common.np2Tensor(*lrs,  rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
            
            return torch.stack(lr_tensors), None, None, filenames
        else:
            if self.args.process:
                lrs, hrs, kernels, filenames = self._load_file_from_loaded_data(idx)
            else:
                lrs, hrs, kernels, filenames = self._load_file(idx)
            
            if len(lrs.shape) == 4:
                b, ih, iw, _ = lrs.shape
            else:
                lrs = np.array([np.expand_dims(lr, axis=2) for lr in lrs])
                b, ih, iw, _ = lrs.shape
                #print(lrs.shape)

            ip = self.args.patch_size

            ix = random.randrange(0, iw - ip + 1)
            iy = random.randrange(0, ih - ip + 1)

            patches = [self.get_patch(lr, hr, ix, iy) for lr, hr in zip(lrs, hrs)]
            lrs = np.array([patch[0] for patch in patches])
            hrs = np.array([patch[1] for patch in patches])
            kernels = np.array([kernel for kernel in kernels])
            lrs = np.array(common.set_channel(*lrs, n_channels=self.args.n_colors))
            hrs = np.array(common.set_channel(*hrs, n_channels=self.args.n_colors))
            lr_tensors = common.np2Tensor(*lrs,  rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
            hr_tensors = common.np2Tensor(*hrs,  rgb_range=self.args.rgb_range, n_colors=self.args.n_colors)
            
            if self.args.pca_input:
                kernel_tensors = torch.from_numpy(kernels).float()
            else:
                kernels = np.reshape(kernels, (kernels.shape[0], kernels.shape[1], kernels.shape[2], 1))
                kernel_tensors = common.kernel2Tensor(*kernels,  rgb_range=self.args.rgb_range, n_colors=self.args.n_colors,norm=False)
                kernel_tensors = torch.stack(kernel_tensors)
            return torch.stack(lr_tensors), torch.stack(hr_tensors), kernel_tensors, filenames

    def __len__(self):
        return sum(self.n_frames_video) - (self.n_seq - 1) * len(self.n_frames_video)

    def _get_index(self, idx):
        return idx

    def _find_video_num(self, idx, n_frame):
        for i, j in enumerate(n_frame):
            if idx < j:
                return i, idx
            else:
                idx -= j

    def _load_file(self, idx):
        """
        Read image from given image directory
        Return: n_seq * H * W * C numpy array and list of corresponding filenames
        """
        if self.args.real:
            n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
            video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
            f_lrs = self.images_lr[video_idx][frame_idx:frame_idx+self.n_seq]
            filenames = [os.path.split(os.path.dirname(file))[-1] + '.' + os.path.splitext(os.path.basename(file))[0] for file in f_lrs]
            lrs = np.array([imageio.imread(lr_name) for lr_name in f_lrs])
            return lrs, None, None, filenames

        else:
            n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
            #print(n_poss_frames, self.n_frames_video)
            video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
            f_hrs = self.images_hr[video_idx][frame_idx:frame_idx+self.n_seq]
            f_lrs = self.images_lr[video_idx][frame_idx:frame_idx+self.n_seq]
            f_kernels = self.images_kernels[video_idx][frame_idx:frame_idx+self.n_seq]
            filenames = [os.path.split(os.path.dirname(file))[-1] + '.' + os.path.splitext(os.path.basename(file))[0] for file in f_hrs]
            hrs = np.array([imageio.imread(hr_name) for hr_name in f_hrs])
            lrs = np.array([imageio.imread(lr_name) for lr_name in f_lrs])
            #print(np.shape(lrs))
            kernels = np.array([np.load(kernel_name) for kernel_name in f_kernels])
            return lrs, hrs, kernels, filenames

    def _load_file_from_loaded_data(self, idx):
        if self.args.real:
            idx = self._get_index(idx)
            n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
            video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
            f_lrs = self.images_lr[video_idx][frame_idx:frame_idx+self.n_seq]
            lrs = self.data_lr[video_idx][frame_idx:frame_idx+self.n_seq]
            filenames = [os.path.split(os.path.dirname(file))[-1] + '.' + os.path.splitext(os.path.basename(file))[0] for file in f_lrs]
            return lrs, None, None, filenames
        else:
            idx = self._get_index(idx)
            n_poss_frames = [n - self.n_seq + 1 for n in self.n_frames_video]
            video_idx, frame_idx = self._find_video_num(idx, n_poss_frames)
            f_hrs = self.images_hr[video_idx][frame_idx:frame_idx+self.n_seq]
            hrs = self.data_hr[video_idx][frame_idx:frame_idx+self.n_seq]
            lrs = self.data_lr[video_idx][frame_idx:frame_idx+self.n_seq]
            kernels = self.data_kernel[video_idx][frame_idx:frame_idx+self.n_seq]
            filenames = [os.path.split(os.path.dirname(file))[-1] + '.' + os.path.splitext(os.path.basename(file))[0] for file in f_hrs]
            return lrs, hrs, kernels, filenames

    def get_patch(self, lr, hr, ix, iy):
        """
        Returns patches for multiple scales
        """
        scale = self.scale

        ih, iw = lr.shape[:2]
        ih -= ih % 4
        iw -= iw % 4
        lr = lr[:ih, :iw]
        if hr is not None:
            hr = hr[:ih * scale, :iw * scale]

        if self.args.mem_opt:
            h = ih//4
            w = iw//4
            lr = lr[h:h*3, w:w*3]
            if hr is not None:
                hr = hr[h*scale:h*scale*3, w*scale:w*scale*3]
        return lr, hr
