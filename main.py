import argparse
import collections
import torch
from time import time
import data_loader as my_data
import model.loss as my_loss
import model.model as my_model
from parse_config import *
from trainer import *
from utils import *
import random

def main(args,config):
    # data loaders
    data_loader = config.init_obj('data_loader',my_data)
    eval_data_loader = data_loader.split_validation()

    # model
    model = config.init_obj('arch',my_model)

    # loss
    loss_class = getattr(my_loss,config["loss"]["type"])
    criterion = config.init_obj('loss',my_loss,cls_num_list=data_loader.cls_num_list,u_value=args.u_value)

    # optimizer
    optimizer = config.init_obj('optimizer',torch.optim,model.parameters())

    # learning rate
    config_lr = config["lr_scheduler"]["args"]
    lr_decay,warmup_epoch = config_lr["lr_decay"],config_lr["warmup_epoch"]

    def lr_lambda(epoch):
        if epoch>=config_lr["stage2"]:
            lr = lr_decay*lr_decay
        elif epoch>=config_lr["stage1"]:
            lr = lr_decay
        else:
            lr = 1
        if epoch<warmup_epoch:
            lr = lr*float(1+epoch)/warmup_epoch
        return lr

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)

    trainer = Trainer(
        model               = model             ,
        criterion           = criterion         ,
        opt                 = optimizer         ,
        args                = args              ,
        config              = config            ,
        data_loader         = data_loader       ,
        eval_data_loader    = eval_data_loader  ,
        lr_scheduler        = lr_scheduler
    )
    trainer.run()

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c','--config',type=str,default='configs/cifar100_lt.json')
    args.add_argument('-g','--gpu',type=int,default=0)
    args.add_argument('-s','--seed',type=int)
    args.add_argument('-u','--u-value',type=float,default=-1)

    args = args.parse_args()
    config = ConfigParser.from_args(args)

    random_seed_setup(args.seed)
    start = time()
    main(args,config)
    end = time()

    minute = (end-start)/60
    if minute<=60:
        print(f'Finished in {minute:.1f} min')
    else:
        print(f'Finished in {minute/60:.1f} h')
