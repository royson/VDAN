import argparse
import template

parser = argparse.ArgumentParser(description='Parameters for VideoSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--gpu_ids', type=str, default='0',
                    help='id of gpus')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../Dataset',
                    help='dataset directory')
parser.add_argument('--dir_data_test', type=str, default='../../Dataset',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set5',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-90/91-100',
                    help='train/test data range')
parser.add_argument('--process', action='store_true',
                    help='if True, load all dataset at once at RAM')
parser.add_argument('--scale', type=str, default=4,
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=100,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--post_fix', type=str, default='*.png',
                    help='post_fix for data')
parser.add_argument('--mem_opt', default=False, action='store_true',
                    help='activate for memory optimization on huge data')

# Video SR parameters
parser.add_argument('--n_sequence', type=int, default=3,
                    help='length of image sequence per video')
parser.add_argument('--n_frames_per_video', type=int, default=30,
                    help='number of frames per video to load')


# Model specifications
parser.add_argument('--model', default='VDAN',
                    help='model name')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')


# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')


# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--lambd', type=float, default=0.0005,
                    help='coefficient for modified huber loss')
parser.add_argument('--beta', type=float, default=0.005,
                    help='coefficient for motioncompensation mse loss')
parser.add_argument('--loss', type=str, default='MSE',
                    help='loss function string')

# Log specifications
parser.add_argument('--save', type=str, default='save_path',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', action='store_true',
                    help='resume from the latest if true')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_images', default=False, action='store_true',
                    help='save images')

# kernel training
parser.add_argument('--test_kernel_path', type=str, default='kernel_blur')
parser.add_argument('--test_blur_path', type=str, default='kernel_GT')
parser.add_argument('--pca_input', action='store_true', help='input type of kernel')
parser.add_argument('--dir_hr',type=str, default='GT')
parser.add_argument('--data_root', type=str, default='REDS4')
parser.add_argument('--real', default=False, action='store_true',
                    help='for real image inference')

parser.add_argument('--corrector', type=str, default='CORRECTOR',
                    help='Model for correct SR results from mismatch kernel')
parser.add_argument('--predictor', type=str, default='TKENOSFT',
                    help='Model for predict LR kernel')
parser.add_argument('--corrector_path', type=str, default='.',
                    help='pre-train for corrector')
parser.add_argument('--predictor_path', type=str, default='.',
                    help='pre-train for predictor')

parser.add_argument('--noise', default=False, action='store_true',
                    help='add noise for pre-training')
parser.add_argument('--gt_ref', default=False, action='store_true',
                    help='using ground turth pre kernel as ref')
parser.add_argument('--use_cor', default=True, action='store_false',
                    help='using corrector when predict')
parser.add_argument('--fix_kernel', default=False, action='store_true',
                    help='using fix kernel for inference')
parser.add_argument('--k_size', type=int, default=13,
                    help='kernel size for estimation')
parser.add_argument('--code_size', type=int, default=13,
                    help='pca kernel size for estimation')
parser.add_argument('--dan_iter', type=int, default=4,
                    help='iteration that DAN running')
parser.add_argument('--real_kernel', type=float, default=0.0,
                    help='percentage of using real world kernel for training')
parser.add_argument('--real_kernel_path', type=str, default='../dataset/REDS4/sth_kernel',
                    help='path of real world kernel')
parser.add_argument('--cloud_save', type=str, default='.',
                    help='path to save in cloud server')
parser.add_argument('--converty', default=True, action='store_false',
                    help='convert y to calc psnr and ssim')
args = parser.parse_args()
template.set_template(args)

if args.epochs == 0:
    args.epochs = 1e8

#Force batch size =1 for best memo usage
if args.mem_opt:
    args.batch_size = 1

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
