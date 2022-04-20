import pdb
import os
import sys
# sys.path.remove('/home/brianyao/Documents/intention2021icra')
# sys.path.append(os.path.realpath('.'))
import torch
from torch import nn #optim
from torch.nn import functional as F

import pickle as pkl
from datasets import make_dataloader
from bitrap.modeling import make_model
from bitrap.engine import build_engine



from bitrap.utils.logger import Logger
import logging

import argparse
from configs import cfg
from termcolor import colored 
import yaml

import pdb

def main(cfg,cfg_zed):
    # build model, optimizer and scheduler
    model = make_model(cfg)
    model = model.to(cfg.DEVICE)
    CKPT_DIR = "ETH/bitrap_np_eth.pth"
    if os.path.isfile(CKPT_DIR):#cfg.CKPT_DIR):
        model.load_state_dict(torch.load(CKPT_DIR))#cfg.CKPT_DIR
        print(colored('Loaded checkpoint:{}'.format(CKPT_DIR), 'blue', 'on_green')) #cfg.CKPT_DIR
    else:
        print(colored('The cfg.CKPT_DIR id not a file: {}'.format(CKPT_DIR), 'green', 'on_red'))#cfg.CKPT_DIR
    
    if cfg.USE_WANDB:
        logger = Logger("MPED_RNN",
                        cfg,
                        project = cfg.PROJECT,
                        viz_backend="wandb"
                        )
    else:
        logger = logging.Logger("MPED_RNN")
    
    # # get dataloaders
   # test_dataloader = make_dataloader(cfg, 'test')
    
    if hasattr(logger, 'run_id'):
        run_id = logger.run_id
    else:
        run_id = 'no_wandb'
    _, _, inference = build_engine(cfg)
    
    #inference(cfg, 0, model, test_dataloader, cfg.DEVICE, logger=logger, eval_kde_nll=True, test_mode=True)
    inference(cfg,cfg_zed, 0, model, cfg.DEVICE, logger=logger, eval_kde_nll=True, test_mode=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument(
        "--config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    print("Configuring: ","configs/bitrap_np_ETH.yml")#args.config_file)
    cfg.merge_from_file("configs/bitrap_np_ETH.yml")#)args.config_file)
    cfg.merge_from_list(args.opts)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open("config_latest.yml", "r") as ymlfile:
        cfg_zed = yaml.safe_load(ymlfile)

    for section in cfg_zed:
        print(section, "  : ",cfg_zed[section])

    main(cfg,cfg_zed)




