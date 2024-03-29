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
import torchvision.transforms as TT
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
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
from torchreid.adabn import bn_update
#from torchreid.utils.rerank import re_ranking
from torchreid.utils.rerank_luo import re_ranking
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
import tqdm
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

    norm_mean_query = [0.351,0.367,0.356]
    norm_std_query = [0.124,0.150,0.160]

    norm_mean_gallery = [0.252,0.288,0.286]
    norm_std_gallery = [0.142,0.167,0.188]

    # norm_mean =  [0.228,0.268,0.264]
    # norm_std =  [0.144,0.166,0.190]
    # transform_train = T.Compose([
    #     T.Random2DTranslation(config.height, config.width, p=0.0),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     T.Normalize(mean=norm_mean, std=norm_std),
    #     T.RandomErasing(),
    # ])

    transform_train = T.Compose([
        TT.Resize([config.height, config.width]),
        TT.RandomHorizontalFlip(p=0.5),
        TT.Pad(10),
        TT.RandomCrop([config.height, config.width]),
        TT.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        TT.transforms.RandomAffine(0, translate=None, scale=[0.9, 1.1], shear=None, resample=False, fillcolor=128),
       # T.RandomPatch(),
        TT.ToTensor(),
        TT.Normalize(mean=norm_mean, std=norm_std),
        T.RandomErasing()
    ])

    pin_memory = True if use_gpu else False
    trainloader = DataLoader(
        ImageDataset(dataset.train, transform=transform_train),
        sampler=RandomIdentitySampler(dataset.train, config.train_batch, config.num_instance),
        batch_size=config.train_batch, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=True,
    )



    transform_test = T.Compose([
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    transform_test_query = T.Compose([
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    transform_test_gallery = T.Compose([
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])



    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test_query),
        batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
        pin_memory=pin_memory, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test_gallery),
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
    

    if config.adabn:
        print("==> using adabn for all bn layers")
        bn_update(model,trainloader,cumulative = 0)


    if config.evaluate:
        print("Evaluate only")
        distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)
        #distmat = getBestRerank(model, queryloader, galleryloader, use_gpu, return_distmat=True)
        
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

def scale(x, scale_factor, interpolation="nearest", align_corners=None):
    """scale batch of images by `scale_factor` with given interpolation mode"""
    h, w = x.shape[2:]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    return F.interpolate(
        x, size=(new_h, new_w), mode=interpolation, align_corners=align_corners
    )

def center_crop(x, crop_h, crop_w):
    """make center crop"""

    center_h = x.shape[2] // 2
    center_w = x.shape[3] // 2
    half_crop_h = crop_h // 2
    half_crop_w = crop_w // 2

    y_min = center_h - half_crop_h
    y_max = center_h + half_crop_h + crop_h % 2
    x_min = center_w - half_crop_w
    x_max = center_w + half_crop_w + crop_w % 2

    return x[:, :, y_min:y_max, x_min:x_max]

def extract_feature(config, dataloader, model, use_gpu, flip=False):
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
            
            if config.TTA:
                imgs = center_crop(imgs, config.height - 20, config.width - 20)
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
    
    with open(os.path.join(config.save_dir, 'test_rematch.json'), 'w') as fw:
        json.dump(result_dict, fw)

    return 


def aqe_func_gpu(all_feature,k2,alpha,len_slice = 1000):
    all_feature = F.normalize(all_feature, p=2, dim=1)
    gpu_feature = all_feature.cuda()
    T_gpu_feature = gpu_feature.permute(1,0)
    all_feature = all_feature.numpy()

    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)

    all_features = []


    for i in range(n_iter):
        # cal sim by gpu
        sims = torch.mm(gpu_feature[i*len_slice:(i+1)*len_slice], T_gpu_feature)
        sims = sims.data.cpu().numpy()
        for sim in sims:
            initial_rank = np.argpartition(-sim,range(1,k2+1)) # 1,N
            # initial_rank = np.argpartition(-sim,k2) # 1,N
            weights = sim[initial_rank[:k2]].reshape((-1,1)) # k2,1
            # weights /= np.max(weights)
            weights = np.power(weights,alpha)
        
            all_features.append(np.mean(all_feature[initial_rank[:k2],:]*weights,axis=0))


    all_feature = np.stack(all_features,axis=0)

    all_feature = torch.from_numpy(all_feature)
    all_feature = F.normalize(all_feature, p=2, dim=1)
    return all_feature


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    qf, q_pids, q_camids = extract_feature(config, queryloader, model, use_gpu, flip=True)
    gf, g_pids, g_camids = extract_feature(config, galleryloader, model, use_gpu, flip=True)
    m, n = qf.size(0), gf.size(0)
   
    if config.aqe:
        all_feature = np.concatenate([qf, gf], axis = 0)
        k2 = 4
        alpha = 3.0
        all_feature = torch.from_numpy(all_feature)
        print("==>using weight query expansion k2: {} alpha: {}".format(k2,alpha))
        all_feature = aqe_func_gpu(all_feature,k2,alpha,len_slice = 2000)
        
        qf = all_feature[:m]
        gf = all_feature[m:]

    
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())
    # distmat = distmat.numpy()


    if config.rerank:
        print('Applying person re-ranking ...')
        # distmat_qq = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
        #         torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        # distmat_qq.addmm_(1, -2, qf, qf.t())
        # distmat_qq = distmat_qq.numpy()

        # distmat_gg = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
        #         torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
        # distmat_gg.addmm_(1, -2, gf, gf.t())
        # distmat_gg = distmat_gg.numpy()

        # distmat = re_ranking(distmat, distmat_qq, distmat_gg)
        #distmat = re_ranking(qf, gf, 30, 4, 0.8)
        distmat = re_ranking(qf, gf, 10, 5, 0.8)


    clusters = {}

    clusters['query_path'] = q_camids
    clusters['gallery_path'] = g_camids

    clusters['query_feat'] = qf
    clusters['gallery_feat'] = gf

    clusters['dist_mat'] = distmat
    
    torch.save(clusters, './result_for_ensemble_C/submission_example_A.pth'.replace('submission_example_A', 'allData1026_' + args.load_weights.split('/')[-2]))

    print("Get Result")
    eval(distmat, q_pids, g_pids, q_camids, g_camids)

    print(" ----- Finished -----")
    return distmat



def getBestRerank(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    qf_, q_pids, q_camids = extract_feature(config, queryloader, model, use_gpu, flip=True)
    gf_, g_pids, g_camids = extract_feature(config, galleryloader, model, use_gpu, flip=True)
    m, n = qf_.size(0), gf_.size(0)


    k2_list = list(range(2, 12))
    alpha_list = list(range(2,7))

    for i in k2_list:
        for j in alpha_list:
            
            print(str(i) + ' ----- ' + str(j))
            print("Computing CMC and mAP")

            if config.aqe:
                all_feature = np.concatenate([qf_, gf_], axis = 0)
                k2 = i
                alpha = j
                all_feature = torch.from_numpy(all_feature)
                print("==>using weight query expansion k2: {} alpha: {}".format(k2,alpha))
                all_feature = aqe_func_gpu(all_feature,k2,alpha,len_slice = 2000)
                
                qf = all_feature[:m]
                gf = all_feature[m:]

            if config.rerank:
                print('Applying person re-ranking ...')
                # distmat_qq = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                #         torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
                # distmat_qq.addmm_(1, -2, qf, qf.t())
                # distmat_qq = distmat_qq.numpy()

                # distmat_gg = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                #         torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
                # distmat_gg.addmm_(1, -2, gf, gf.t())
                # distmat_gg = distmat_gg.numpy()

                # distmat = re_ranking(distmat, distmat_qq, distmat_gg)
                #distmat = re_ranking(qf, gf, 30, 4, 0.8)
                distmat = re_ranking(qf, gf, 10, 5, 0.8)

            cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
            print("Results ----------")
            print("mAP: {:.1%}".format(mAP))
            print("CMC curve")
            for r in ranks:
                print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
            print("------------------")
    # if config.rerank:
    #     print('Applying person re-ranking ...')
    #     # distmat_qq = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
    #     #         torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
    #     # distmat_qq.addmm_(1, -2, qf, qf.t())
    #     # distmat_qq = distmat_qq.numpy()

    #     # distmat_gg = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
    #     #         torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
    #     # distmat_gg.addmm_(1, -2, gf, gf.t())
    #     # distmat_gg = distmat_gg.numpy()

    #     # distmat = re_ranking(distmat, distmat_qq, distmat_gg)
 
    #     k1 = list(range(6, 31))
    #     k2 = list(range(2,7))
    #     alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #     for i in alpha:
    #         for j in k2:
    #             for k in k1:
    #                 distmat = re_ranking(qf, gf, k, j, i)
    #                 print(str(k) + ' ----- ' + str(j) + ' ----- ' + str(i))
    #                 print("Computing CMC and mAP")
    #                 cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    #                 print("Results ----------")
    #                 print("mAP: {:.1%}".format(mAP))
    #                 print("CMC curve")
    #                 for r in ranks:
    #                     print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    #                 print("------------------")


    print(" ----- Finished -----")
    return distmat


if __name__ == '__main__':
    main()
