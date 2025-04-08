import pdb
import os
import time
import numpy as np
import torch
import logging
import faulthandler
faulthandler.enable()
from Config import *
import json
import argparse
from module.gpu_config import init_distributed_mode
from module.load_tokenizer import load_tokenizer
from module.DataManager import DataManager
from module.Trainer import Trainer
import torch.distributed as dist

def print_all_attributes(obj):
    # 遍历对象的所有属性（包括继承的）
    for attr in dir(obj):
        # 排除特殊属性（以双下划线开头的）
        if not attr.startswith("__"):
            # 获取属性的值
            value = getattr(obj, attr)
            print(f"{attr}: {value}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',required=True)

    parser.add_argument('--config', default='Config')
    parser.add_argument('--mode', default='default')

    parser.add_argument('--path_model_save',default='default')
    parser.add_argument('--path_output_file',default='')

    parser.add_argument('--distributed',action='store_true')

    parser.add_argument('--train_dataset',default='default')
    parser.add_argument('--val_dataset',default='default')

    parser.add_argument('--resume_from',default='default')
    args = parser.parse_args()
    config = eval(args.config)()
    
    print(f"args.distributed: {args.distributed}")
    config.distributed=args.distributed
    if config.distributed:
        config.world_size=int(os.environ["WORLD_SIZE"])
    config.mode=args.mode

    if args.path_model_save!='default':
        config.path_model_save=args.path_model_save
    # if args.path_output_file!='default':
    config.path_output_file=args.path_output_file

    if args.train_dataset!='default':
        config.train_dataset=args.train_dataset
    if args.val_dataset!='default':
        config.val_dataset=args.val_dataset


    if args.resume_from!='default':
        config.resume_from=args.resume_from

    # 设置随机种子，保证结果每次结果一样
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    print_all_attributes(config)
    # if (args.distributed and int(os.environ["RANK"])==0) or (args.distributed==0):
    #     config_dict = vars(config)
    #     print(config_dict)

    #     # 将字典数据 dump 到文件
    #     with open(f'{args.dir}/config.json', 'w') as f:
    #         json.dump(config_dict, f, indent=4)

    #ddp设置
    init_distributed_mode(config)
    #tokenizer设置
    config.tokenizer=load_tokenizer(config)

    # 数据处理
    print('read data...')
    dm = DataManager(config)

    # 模式
    if config.mode == 'train':
        # 获取数据
        """print('data process...')
        train_loader = dm.get_dataset(data_type='train_pred_0611')
        valid_loader = dm.get_dataset(data_type='dev_pred_0611')
        test_loader = dm.get_dataset(data_type='test_pred_0611')"""
        print('data process...')
        if not config.train_dataset is None:
            train_loader = dm.get_dataset(file=config.train_dataset,data_type='train')
        else:
            train_loader=None
        if not config.val_dataset is None:
            valid_loader = dm.get_dataset(file=config.val_dataset,data_type='val')
        else:
            valid_loader=None
        if not config.test_dataset is None:
            test_loader = dm.get_dataset(file=config.test_dataset,data_type='test')
        else :
            test_loader = None 
        # 训练
        trainer = Trainer(config, train_loader, valid_loader, test_loader)
        trainer.train()
    elif config.mode == 'infer':
        if not config.val_dataset is None:
            valid_loader = dm.get_dataset(file=config.val_dataset,data_type=config.val_dataset)
        else:
            raise Exception
        trainer = Trainer(config, train_loader=None, valid_loader=None, test_loader=None)
        trainer.evaluate(data=valid_loader) 
    elif config.mode=='embedding':
        trainer = Trainer(config, train_loader=None, valid_loader=None, test_loader=None)
        text="""
        .#ootd# ., BANNED, dedmemess_27, You have ben banned for being abusive or ffensive to other, players over voice or text chat., You were banned from online play by Psyonix., 72 hour(s)1minute(s)remaining, CHAT LOG:, and u abitch, stfu, stfu, nigger, OK, RTY, BANNED, dedmemess 27, You have been banned for being abusive or fensive to other, players over voice or text chat., You were banned from online play by Psyonix., 72hour(s1minutesremaining, CHAT LOG:, and u abitch, stfu, stfu, nigger, OK, TY, BANNED, dedmemess 27, You have been banned for being abusive or ffensive to other, players over voice or text chat., You were banned from online play by Psyonix, 72hour(s)1minutesremaining, CHATLOG:, and u abitch, stfu, stfu, nigger, OK, RTY, Alli said was gg, BANNED, dedmemess 27, You have been banned for being abusive or ffensive to other, players over voice or text chat., You were banned from online play by Psyonix., 72hour(s1minute(sremaining., CHat lOG:, and u abitch, stfu, stfu, nigger, OK, ARTY, BANNED, dedmemess 27, You have been banned for being abusive or ffensive to other, players over voice or text chat., You were banned from online play by Psyonix., 72hour(s)1minute(sremaining., CHAT LOG:, and u abitch, stfu, stfu, nigger, OK, BANNED, dedmemess_27, You have been banned for being abusive or ffensive to other, players over voice or text chat, You were banned from online play by Psyonix, 72 hour(s)1minute(sremaining, CHAT LOG, and u abitch, stfu, stfu, nigger, OK, RTY
        """
        trainer.get_embedding(text=text)      


    if config.distributed:
        dist.destroy_process_group()