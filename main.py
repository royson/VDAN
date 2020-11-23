import torch

import data
import model
from option import args
from trainer_vdanpcdtsa import Trainer_VDANPCDTSA
from logger import logger
import copy

torch.manual_seed(args.seed)
chkp = logger.Logger(args)
if args.task == 'KernelSR':
    print("Selected task: KernelSR")
    loader = data.Data(args)
    model = model.Model(args, chkp)
    t = Trainer_VDANPCDTSA(args, loader, model, chkp)

    while not t.terminate():
        t.train()
        t.test()
else:
    print('Please Enter Appropriate Task Type!!!')

chkp.done()
