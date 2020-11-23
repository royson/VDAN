import torch

import data
import model
from option import args
from trainer import Trainer_VDAN
from logger import logger
import copy

torch.manual_seed(args.seed)
chkp = logger.Logger(args)
loader = data.Data(args)
model = model.Model(args, chkp)
t = Trainer_VDAN(args, loader, model, chkp)
while not t.terminate():
    t.train()
    t.test()

chkp.done()
