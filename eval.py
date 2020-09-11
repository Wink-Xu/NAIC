from __future__ import print_function
from __future__ import division

import os
import sys
pwd = os.getcwd()
sys.path.append(os.path.join(pwd, 'configs'))

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
from torchreid import models
from torchreid.losses import CrossEntropyLabelSmooth, DeepSupervision, MGSupervision, TripletLoss
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import set_bn_to_eval, count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.utils.rerank import re_ranking
from torchreid.eval_metrics import evaluate
from torchreid.samplers import RandomIdentitySampler
from torchreid.optimizers import init_optim
from torchreid.lr_scheduler import WarmupMultiStepLR
import torchvision
from IPython import embed
import ipdb as pdb
from configs.merge_config import merge_config
import importlib
import json

parser = argparse.ArgumentParser(description='Train image model')


parser.add_argument('--config_type', default='config', type=str,
                    help="config type")
parser.add_argument('--dataset', default='', type=str,
                    help="dataset")
parser.add_argument('--max-epoch', default=60, type=int,
                    help="maximum epochs to run")
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

    sys.stdout = Logger(osp.join(config.save_dir, 'log_test.txt'))

    print("==========\nArgs:{}\n==========".format(config))

    if use_gpu:
        print("Currently using GPU {}".format(config.gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(config.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(config.dataset))
    dataset = data_manager.init_imgreid_dataset(
        root=config.root, name=config.dataset, split_id=config.split_id
    )

    norm_mean = [0.485, 0.456, 0.406] + [0.0]*config.mask_num
    norm_std = [0.229, 0.224, 0.225] + [1.0]*config.mask_num
    transform_train = T.Compose([
        T.Random2DTranslation(config.height, config.width, p=0.0),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
        T.RandomErasing(),
    ])

    transform_test = T.Compose([
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test),
        batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test),
        batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    print("Initializing model: {}".format(config.arch))
    model = models.init_model(name=config.arch, num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu, config = config)
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

    if config.resume:
        if check_isfile(config.resume):
            checkpoint = torch.load(config.resume)
            pretrain_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            for k, v in pretrain_dict.items():
                if not k in model_dict.keys():
                    print('Not Used Layer:', k)
                    continue
                if model_dict[k].shape == v.shape:
                    model_dict[k] = v
                else:
                    pid = int(k.split('.')[1][-1]) - 1
                    model_dict[k].zero_()
                    model_dict[k][:,:,pid:pid+1,:] = v
                    print(k)
            model.load_state_dict(model_dict)
            config.start_epoch = checkpoint['epoch']
            rank1 = checkpoint['rank1']
            print("Loaded checkpoint from '{}'".format(config.resume))
            print("- start_epoch: {}\n- rank1: {}".format(config.start_epoch, rank1))
            for epoch in range(0, config.start_epoch):
                scheduler.step()
            model.eval()
            if config.onnx_export:
                torch_out = torch.onnx._export(model, torch.rand(1, 3, config.height, config.width), 
                                       osp.join(config.save_dir, config.onnx_export), export_params=True)
                #return

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    if config.evaluate:
        print("Evaluate only")
        distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)
        if config.vis_ranked_res:
            visualize_ranked_results(
                distmat, dataset,
                save_dir=osp.join(config.save_dir, 'ranked_results'),
                topk=8,
            )
        return


def class_preds(outputs):
    softmax = nn.Softmax(dim=1)
    if isinstance(outputs, tuple):
        score = torch.zeros(outputs[0].size(), device=outputs[0].device)
        for op in outputs:
            score += softmax(op)
    else:
        score = softmax(outputs)
    _, preds = torch.max(score.data, 1)
    return preds

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

            features = model(imgs, pids)

            if flip:
                if isinstance(imgs, tuple):
                    imgs, keyps = imgs[0], imgs[1]
                    imgs = fliplr(imgs, use_gpu)
                    imgs = (imgs, keyps) 
                else:
                    imgs = fliplr(imgs, use_gpu)
                
                features += model(imgs, pids)

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

def eval(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 200):
    """Evaluation 
       Get result of json format
    """
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)
    imgName_rank = g_camids[indices[:,:200]]
    result_dict = {}
    for i in range(num_q):
        result_dict[q_camids[i]] = list(imgName_rank[i])
    
    with open(os.path.join(config.save_dir, 'test.json'), 'w') as fw:
        json.dump(result_dict, fw)

    return 




def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    qf, q_pids, q_camids = extract_feature(queryloader, model, use_gpu, flip=True)
    gf, g_pids, g_camids = extract_feature(galleryloader, model, use_gpu, flip=True)
        
    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    if config.rerank:
        print('Applying person re-ranking ...')
        distmat_qq = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        distmat_qq.addmm_(1, -2, qf, qf.t())
        distmat_qq = distmat_qq.numpy()

        distmat_gg = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
        distmat_gg.addmm_(1, -2, gf, gf.t())
        distmat_gg = distmat_gg.numpy()

        distmat = re_ranking(distmat, distmat_qq, distmat_gg)


    print("Get Result")
    eval(distmat, q_pids, g_pids, q_camids, g_camids)
    
    print(" ----- Finished -----")
    return distmat


if __name__ == '__main__':
    main()
