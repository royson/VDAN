import os
import math
import time
import imageio
import decimal
import random
import numpy as np
from scipy import misc
import skimage.color as sc
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

import utils
import os
import time

class Trainer_VDAN:
    def __init__(self, args, loader, my_model, ckp):
        self.args = args
        self.scale = args.scale
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.optimizer = self.make_optimizer()
        self.scheduler = self.make_scheduler()
        self.ckp = ckp
        if self.args.loss == "L1":
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.MSELoss()
        if args.load != '.':
            self.optimizer.load_state_dict(torch.load(os.path.join(ckp.dir, 'optimizer.pt')))
            for _ in range(len(ckp.psnr_log)):
                self.scheduler.step()
        
        pca_matrix_path = os.path.join('experiment', 'pca_matrix_{}_codelen_{}.pth'.format(self.args.scale, self.args.code_size))
        if not os.path.exists(pca_matrix_path):
            batch_ker = utils.random_batch_kernel(batch=30000, l=self.args.k_size, sig_min=0.6, sig_max=5.0, rate_iso=0.1, scaling=self.args.scale, tensor=False)
            b = np.size(batch_ker, 0)
            batch_ker = batch_ker.reshape((b, -1))
            print('calculating PCA projection matrix...')
            self.pca_matrix = utils.PCA(batch_ker, k=self.args.code_size).float() 
            torch.save(self.pca_matrix, pca_matrix_path)
        else:
            print('loading PCA projection matrix...')
            self.pca_matrix = torch.load(pca_matrix_path, map_location=lambda storage, loc: storage)


    def set_loader(self, new_loader):
        self.loader_train = new_loader.loader_train
        self.loader_test = new_loader.loader_test

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        return optim.Adam(self.model.parameters(), **kwargs)

    def make_scheduler(self):
        kwargs = {'step_size': self.args.lr_decay, 'gamma': self.args.gamma}
        return lrs.StepLR(self.optimizer, **kwargs)

    def train(self):
        self.epoch = self.scheduler.last_epoch + 1
        self.scheduler.step()
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(self.epoch, decimal.Decimal(lr)))

        self.model.train()
        self.ckp.start_log()
        for batch, (lr, hr, filename) in enumerate(self.loader_train):
            #lr: [batch_size, n_seq, 3, patch_size, patch_size]
            filename = filename[len(filename)//2][0] + '.test'
            #print(filename)
            if self.args.n_colors == 1 and lr.size()[2] == 3:
                lr = lr[:, :, 0:1, :, :]
                hr = hr[:, :, 0:1, :, :]


            # Divide LR frame sequence [N, n_sequence, n_colors, H, W] -> N * [1, n_sequence, n_colors, H, W]
            lr = list(torch.split(lr, 1, dim = 0))
            hr = list(torch.split(hr, 1, dim = 0))

            # Get current hr batch for final sr results
            center = self.args.n_sequence // 2
            center_hr =  [x[:, center, :, :, :] for x in hr]
            center_hr = [x.to(self.device) for x in center_hr]
            center_hr = torch.cat(center_hr, dim = 0)


            #define preprocess, blurred, downscaled and add noise on HR
            prepro = utils.SRMDPreprocessing(self.args.scale, self.pca_matrix, random=True, para_input=self.args.code_size,
                                            kernel=self.args.k_size, noise=self.args.noise, cuda=True, sig=2.6,
                                            sig_min=0.6, sig_max=5.0, rate_iso=0.1, scaling=self.args.scale,
                                            rate_cln=0.2, noise_high=0.25, n_seq=self.args.n_sequence, real=self.args.real_kernel, real_path=self.args.real_kernel_path)

            

            cur_hr = hr
            # Squeeze on dimension 0 now cur_hr should be N * [n_sequence, n_colors, H, W]
            cur_hr = [torch.squeeze(x, dim = 0) for x in cur_hr]
            # Cat it in first dimension, now it batching as [N*n_sequence, n_colors, H, W]
            cur_hr = torch.cat(cur_hr, dim = 0)
            # Get LR sequence in [N*n_sequence, n_colors, H/scale, W/scale]
            cur_blur, cur_ker_map = prepro(cur_hr)

            B, L = cur_ker_map.size()
            cur_ker_map = cur_ker_map.view((B//self.args.n_sequence, L*self.args.n_sequence))
            cur_hr = cur_hr.to(self.device)
            cur_hr = torch.squeeze(cur_hr, dim = 1)
        
            self.optimizer.zero_grad()

            results = self.model(cur_blur, cur_ker_map)
            total_loss = 0

            sr, temp_srs, est_ker_maps = results

            for i in range(len(est_ker_maps)):
                ker_loss = self.loss(est_ker_maps[i], cur_ker_map)
                sr_loss = self.loss(temp_srs[i], cur_hr)
                total_loss += (ker_loss/self.args.n_sequence)*0.25
                total_loss += (sr_loss/self.args.n_sequence)*0.25

            final_loss = self.loss(sr, center_hr)
            total_loss += final_loss
            total_loss.backward()
            self.ckp.report_log(total_loss.item())
            self.optimizer.step()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : {:.5f}'.format(
                    (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1)))


        self.ckp.end_log(len(self.loader_train))

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        self.ckp.start_log(train=False, key='ssim')
        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, data_pack in enumerate(tqdm_test):
                if self.args.real:
                    lr, filename = data_pack
                else:    
                    lr, hr, kernels, filename = data_pack
                ycbcr_flag = False
                filename = filename[len(filename)//2]
                # lr: [batch_size, n_seq, 3, patch_size, patch_size]
                if self.args.n_colors == 1 and lr.size()[2] == 3:
                    lr = lr[:, :, 0:1, :, :]
                    if not self.args.real:
                        hr = hr[:, :, 0:1, :, :]


                # Divide LR frame sequence [N, n_sequence, n_colors, H, W] -> N * [1, n_sequence, n_colors, H, W]
                # We need seperate on first dimension because we want to keep sequence order when re-concact
                lr = list(torch.split(lr, 1, dim = 0))
                lr = [x.to(self.device) for x in lr]
                lr = [torch.squeeze(x, dim = 0) for x in lr]
                lr = torch.cat(lr, dim = 0)
                if not self.args.real:
                    hr = list(torch.split(hr, 1, dim = 0))
                    center = self.args.n_sequence // 2
                    center_hr =  [x[:, center, :, :, :] for x in hr]
                    center_hr = [x.to(self.device) for x in center_hr]
                    center_hr = torch.cat(center_hr, dim = 0)

                    hr = [x.to(self.device) for x in hr]
                    hr = [torch.squeeze(x, dim = 0) for x in hr]
                    hr = torch.cat(hr, dim = 0)
                cur_kernel_pca = None

                sr, _, _, = self.model(lr, cur_kernel_pca)
                sr = torch.clamp(sr, min=0.0, max=1.0)
                if not self.args.real:
                    PSNR = utils.calc_psnr(self.args, sr, center_hr)
                    SSIM = utils.calc_ssim(self.args, sr, center_hr)
                    self.ckp.report_log(PSNR, train=False)
                    self.ckp.report_log(SSIM, train=False, key='ssim')

                if self.args.save_images and idx_img%30 == 0 or self.args.test_only:

                    if self.args.real:
                        save_list = [sr]
                    else:
                        save_list = [sr]
                    
                    filename = filename[0]
                    self.ckp.save_images(filename, save_list, self.args.scale)

            self.ckp.end_log(len(self.loader_test), train=False)
            self.ckp.end_log(len(self.loader_test), train=False, key='ssim')
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage PSNR: {:.3f} , average SSIM: {:.3f} (Best: {:.3f} @epoch {})'.format(
                                    self.args.data_test, self.ckp.psnr_log[-1], self.ckp.ssim_log[-1],
                                    best[0], best[1] + 1))
            if not self.args.test_only:
                self.ckp.save(self, self.epoch, is_best=(best[1] + 1 == self.epoch))

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
