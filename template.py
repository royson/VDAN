from pathlib import Path
import os
home = str(Path.home())

def set_template(args):
    if args.template == 'KSR_REDS_blur':
        args.test_every = 1000
        args.n_frames_per_video = 99
        args.data_range = '0-250/251-269'
        args.data_train = 'REDS_VIDEO'
        args.process = True
        args.dir_data =  os.path.join(home, 'VDAN/dataset/')
        args.data_test = 'REDS4_Kernel'
        args.dir_data_test =  os.path.join(home, 'VDAN/dataset/')
        if args.test_only:
            args.save = '_'.join((args.template, args.test_kernel_path, args.task, 
                                    'Noise', str(args.noise), 'FIXKERNEL', str(args.fix_kernel),
                                    "SEQ",str(args.n_sequence), args.model))
        else:
            args.save = '_'.join((args.template, args.data_test, args.task, 
                                    'Noise', str(args.noise), 'Real_kernel', str(args.real_kernel), 
                                     "SEQ",str(args.n_sequence), args.model))
    else:
        print('unknow template')