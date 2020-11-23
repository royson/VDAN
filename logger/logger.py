import torch
import imageio
import numpy as np
import os
import datetime
from scipy import misc
import skimage.color as sc
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Logger:
    def __init__(self, args):
        self.args = args
        self.psnr_log = torch.Tensor()
        self.loss_log = torch.Tensor()
        self.psnr_log_iter_0 = torch.Tensor()
        self.psnr_log_iter_1 = torch.Tensor()
        self.psnr_log_iter_2 = torch.Tensor()
        self.psnr_log_iter_3 = torch.Tensor()


        if args.load == '.':
            if args.save == '.':
                args.save = datetime.datetime.now().strftime('%Y%m%d_%H:%M')
            self.dir = 'experiment/' + args.save
        else:
            self.dir = 'experiment/' + args.load
            if not os.path.exists(self.dir):
                args.load = '.'
            else:
                self.loss_log = torch.load(self.dir + '/loss_log.pt')
                self.psnr_log = torch.load(self.dir + '/psnr_log.pt')
                print('Continue from epoch {}...'.format(len(self.psnr_log)))
        
        if args.cloud_save != '.':
            self.dir = os.path.join(args.cloud_save, self.dir)

        if args.reset:
            os.system('rm -rf {}'.format(self.dir))
            args.load = '.'

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
            if not os.path.exists(self.dir + '/model'):
                os.makedirs(self.dir + '/model')
        if not os.path.exists(self.dir + '/result/'+self.args.data_test):
            print("Creating dir for saving images...", self.dir + '/result/'+self.args.data_test)
            os.makedirs(self.dir + '/result/'+self.args.data_test)

        print('Save Path : {}'.format(self.dir))

        open_type = 'a' if os.path.exists(self.dir + '/log.txt') else 'w'
        self.log_file = open(self.dir + '/log.txt', open_type)
        with open(self.dir + '/config.txt', open_type) as f:
            f.write('From epoch {}...'.format(len(self.psnr_log)) + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')

    def save(self, trainer, epoch, is_best):
        if self.args.task == 'IKC':
            trainer.model_C.save(self.dir, is_best, filename='C')
            trainer.model_P.save(self.dir, is_best, filename='P')
            trainer.model_F.save(self.dir, is_best, filename='F')
        else: 
            trainer.model.save(self.dir, is_best)
            torch.save(self.loss_log, os.path.join(self.dir, 'loss_log.pt'))
            torch.save(self.psnr_log, os.path.join(self.dir, 'psnr_log.pt'))
            torch.save(trainer.optimizer.state_dict(), os.path.join(self.dir, 'optimizer.pt'))
            self.plot_loss_log(epoch)
            self.plot_psnr_log(epoch)

    def save_images(self, filename, save_list, scale):
        if self.args.task == 'KernelSR':
            f = filename.split('.')
            filename = '{}/result/{}/{}/{}_'.format(self.dir, self.args.data_test, f[0], f[1].zfill(8))
            if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
            postfix = ['SR']
        for img, post in zip(save_list, postfix):
            img = img[0].data.mul(255 / self.args.rgb_range)
            print(img.shape)
            img = np.transpose(img.cpu().numpy(), (1, 2, 0))
            if img.shape[2] == 1:
                img = img.squeeze(axis=2)
            elif img.shape[2] == 3 and self.args.n_colors == 1:
                img = sc.ycbcr2rgb(img.astype('float')).clip(0, 1)
                img = (255 * img).round().astype('uint8')
            imageio.imwrite('{}{}.png'.format(filename, post), img)
            

    def start_log(self, train=True, key='final'):
        if train:
            self.loss_log = torch.cat((self.loss_log, torch.zeros(1)))
        else:
            if key == 'final':
                self.psnr_log = torch.cat((self.psnr_log, torch.zeros(1)))
            elif key == '0':
                self.psnr_log_iter_0 = torch.cat((self.psnr_log, torch.zeros(1)))
            elif key == '1':
                self.psnr_log_iter_1 = torch.cat((self.psnr_log, torch.zeros(1)))
            elif key == '2':
                self.psnr_log_iter_2 = torch.cat((self.psnr_log, torch.zeros(1)))
            elif key == '3':
                self.psnr_log_iter_3 = torch.cat((self.psnr_log, torch.zeros(1)))
            

    def report_log(self, item, train=True, key='final'):
        if train:
            self.loss_log[-1] += item
        else:
            if key == 'final':
                self.psnr_log[-1] += item
            elif key == '0':
                self.psnr_log_iter_0[-1] += item
            elif key == '1':
                self.psnr_log_iter_1[-1] += item
            elif key == '2':
                self.psnr_log_iter_2[-1] += item
            elif key == '3':
                self.psnr_log_iter_3[-1] += item

    def end_log(self, n_div, train=True, key='final'):
        if train:
            self.loss_log[-1].div_(n_div)
        else:
            if key == 'final':
                self.psnr_log[-1].div_(n_div)
            elif key == '0':
                self.psnr_log_iter_0[-1].div_(n_div)
            elif key == '1':
                self.psnr_log_iter_1[-1].div_(n_div)
            elif key == '2':
                self.psnr_log_iter_2[-1].div_(n_div)
            elif key == '3':
                self.psnr_log_iter_3[-1].div_(n_div)

    def plot_loss_log(self, epoch):
        # epoch = epoch - 1
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('Loss Graph')
        #print(axis, self.loss_log.numpy())
        plt.plot(axis, self.loss_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'loss.pdf'))
        plt.close(fig)

    def plot_psnr_log(self, epoch):
        # epoch = epoch - 1
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title('PSNR Graph')
        plt.plot(axis, self.psnr_log.numpy())
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('PSNR')
        plt.grid(True)
        plt.savefig(os.path.join(self.dir, 'psnr.pdf'))
        plt.close(fig)

    def done(self):
        self.log_file.close()
