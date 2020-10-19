from __future__ import print_function
from __future__ import division


import shutil
import os
import pdb

###################### abstract noLabel img to one directory ###########################

# imagepath = '../../data/NAIC_2020/train/images'
# noLabelPath = '../../data/NAIC_2020/train/noLabelImg'
# withLabelImg = {}
# with open('../../data/NAIC_2020/train/label.txt', 'r') as fr:
#     lines = fr.readlines()
#     for line in lines:
#         withLabelImg[line.split(':')[0]] = 1
#     for i in os.listdir(imagepath):
#         if i not in withLabelImg:
#             imgName = os.path.join(imagepath, i)
#             shutil.copy(imgName, os.path.join(noLabelPath, i))


###################### extract noLabel img feature ###########################

import os
import sys
pwd = os.getcwd()
sys.path.append(os.path.join(pwd, '../configs'))
sys.path.append(os.path.join(pwd, '../'))
import time
import datetime
import argparse
import os.path as osp
import numpy as np
import scipy.io

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from tensorboardX import SummaryWriter
from torchreid import data_manager
from torchreid.dataset_loader import ImageDataset
from torchreid import transforms as T
import torchvision.transforms as TT
from torchreid import models
from torchreid.losses import CrossEntropyLabelSmooth, DeepSupervision, MGSupervision, TripletLoss, WeightedTripletLoss, CenterLoss, OIMLoss
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import set_bn_to_eval, count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.samplers import RandomIdentitySampler
from torchreid.optimizers import init_optim, make_optimizer
from torchreid.lr_scheduler import WarmupMultiStepLR
import torchvision
from IPython import embed
import ipdb as pdb
from configs.merge_config import merge_config
import importlib
from data_manager import DataManager


parser = argparse.ArgumentParser(description='Train image model')


parser.add_argument('--config_type', default='config', type=str,
                    help="config type")
parser.add_argument('--dataset', default='', type=str,
                    help="dataset")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size")
parser.add_argument('--evaluate', action='store_true',
                    help="evaluation only")
parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--debug', action='store_true', help='start debug mode')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--load-weights', type=str, default='',
                    help="load pretrained weights but ignores layers that don't match in size")
parser.add_argument('--vis-ranked-res', action='store_true',
                    help="visualize ranked results, only available in evaluation mode (default: False)")
           

args = parser.parse_args()
config_type = args.config_type
CONFIG = importlib.import_module(config_type)
config = merge_config(CONFIG.config, args)



def main():
    
    torch.manual_seed(config.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_devices
    use_gpu = torch.cuda.is_available()
    if config.use_cpu: use_gpu = False

    print("==========\nArgs:{}\n==========".format(config))

    if use_gpu:
        print("Currently using GPU {}".format(config.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(config.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")
 
    print("Initializing dataset {}".format(args.dataset))
    dataset = DataManager(args.dataset)

    norm_mean = [0.485, 0.456, 0.406] + [0.0]*config.mask_num
    norm_std = [0.229, 0.224, 0.225] + [1.0]*config.mask_num
    
    transform_test = T.Compose([
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    pin_memory = True if use_gpu else False

    dataloader = DataLoader(
        ImageDataset(dataset.dataset, transform=transform_test),
        batch_size=64, shuffle=False, num_workers=4,
        pin_memory=pin_memory, drop_last=False,
    )

    model = models.init_model(name=config.arch, num_classes=dataset.num_pids, loss={'xent'}, use_gpu=use_gpu, config = config)
    print("Model size: {:.3f} M".format(count_num_param(model)))
    print(model)

    if config.load_weights:
        # load pretrained weights but ignore layers that don't match in size
        if check_isfile(config.load_weights):
            checkpoint = torch.load(config.load_weights)
            pretrain_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
            model_dict.update(pretrain_dict)
            model.load_state_dict(model_dict)
            print("Loaded pretrained weights from '{}'".format(config.load_weights))
            rank1 = checkpoint['rank1']
            print("- start_epoch: {}\n- rank1: {}".format(config.start_epoch, rank1))

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    featu, pids, camids = extract_feature(dataloader, model, use_gpu, flip=True)
    save_featu(featu, dataset.dataset)


def save_featu(featu, dataset):
    dim = featu.shape[1]
    for ind, (img,_,_) in enumerate(dataset):
        f = featu[ind]
        src_img = img.replace(args.dataset, args.dataset)
        vec_file = osp.splitext(src_img)[0]+'.vec'
        with open(vec_file, 'w') as fp:
            fp.write('%d '%dim)
            for i in range(dim):
                fp.write('%f '%(f[i]))
            fp.write('\n')
    return

def fliplr(img, use_gpu):
    '''flip horizontal'''
    if use_gpu:
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    else:
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(dataloader, model, use_gpu, flip=False):
    batch_time = AverageMeter()
    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(dataloader):
            if use_gpu:
                if isinstance(imgs, list):
                    imgs = (imgs[0].cuda(), imgs[1].cuda()) 
                else:
                    imgs = imgs.cuda()

            end = time.time()

            if 'lg' in config.arch:
                features = model(imgs, pids)
            elif 'pg'==config.arch[:2]:
                #pdb.set_trace()
                features = model(imgs[0], imgs[1])
            else:
                features = model(imgs)

            if flip:
                if isinstance(imgs, tuple):
                    imgs, keyps = imgs[0], imgs[1]
                    imgs = fliplr(imgs, use_gpu)
                    imgs = (imgs, keyps) 
                else:
                    imgs = fliplr(imgs, use_gpu)
                
                if 'lg' in config.arch:
                    features = model(imgs, pids)
                elif 'pg'==config.arch[:2] :
                    features = model(imgs[0], imgs[1])
                else:
                    features += model(imgs)

            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        print("Extracted features, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))
        print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, config.test_batch))

        qf = qf.squeeze()
        return qf, q_pids, q_camids


if __name__ == '__main__':
    main()
