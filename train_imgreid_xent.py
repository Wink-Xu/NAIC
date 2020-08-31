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
import torchvision.transforms as TT
from torchreid import models
from torchreid.losses import CrossEntropyLabelSmooth, DeepSupervision, MGSupervision, TripletLoss
from torchreid.utils.iotools import save_checkpoint, check_isfile
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.logger import Logger
from torchreid.utils.torchtools import set_bn_to_eval, count_num_param
from torchreid.utils.reidtools import visualize_ranked_results
from torchreid.eval_metrics import evaluate
from torchreid.samplers import RandomIdentitySampler3
from torchreid.optimizers import init_optim
from torchreid.lr_scheduler import WarmupMultiStepLR
import torchvision
from IPython import embed
import ipdb as pdb
from configs.merge_config import merge_config
import importlib

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

    if not config.evaluate:
        assert osp.exists(config.save_dir)==False, '%s already exists'%(config.save_dir)
        sys.stdout = Logger(osp.join(config.save_dir, 'log_train.txt'))
        summary_writer = SummaryWriter(osp.join(config.save_dir, 'tensorboard_log'))
    else:
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

    # transform_train = T.Compose([
    #     T.Random2DTranslation(config.height, config.width, p=0.0),
    #     T.RandomHorizontalFlip(),
    #     TT.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
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
        TT.ToTensor(),
        TT.Normalize(mean=norm_mean, std=norm_std),
        T.RandomErasing()
    ])



    transform_test = T.Compose([
        T.Resize((config.height, config.width)),
        T.ToTensor(),
        T.Normalize(mean=norm_mean, std=norm_std),
    ])

    pin_memory = True if use_gpu else False


    if config.class_balance: 
        trainloader = DataLoader(
            ImageDataset(dataset.train, transform=transform_train),
            sampler=RandomIdentitySampler3(dataset.train, config.train_batch, 4),
            batch_size=config.train_batch, num_workers=config.workers,
            pin_memory=pin_memory, drop_last=True,
        )
    else:
        trainloader = DataLoader(
            ImageDataset(dataset.train, transform=transform_train),
            batch_size=config.train_batch, shuffle=True, num_workers=config.workers,
            pin_memory=pin_memory, drop_last=True,
        )
    print(len(dataset.train), len(trainloader))

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
    model = models.init_model(name=config.arch, num_classes=dataset.num_train_pids, loss={'xent'}, use_gpu=use_gpu)
    print("Model size: {:.3f} M".format(count_num_param(model)))
    print(model)

    if config.label_smooth:
        criterion = CrossEntropyLabelSmooth(num_classes=dataset.num_train_pids, use_gpu=use_gpu)
    else:
        criterion = nn.CrossEntropyLoss()


    if not config.debug:
        optimizer = init_optim(config.optim, model.parameters(), config.lr, config.weight_decay)
        if config.warmup:
            print("---------- using Warmup method -----------")
            scheduler = WarmupMultiStepLR(optimizer, milestones = config.stepsize, gamma=config.gamma, warmup_factor = 0.01,                                                        warmup_iters = 10, warmup_method='linear')
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.stepsize, gamma=config.gamma)
    else:
        dg_stepsize = list(range(20))
        dg_lr = 1e-8
        optimizer = init_optim(config.optim, model.parameters(), dg_lr, config.weight_decay)
      #  scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=dg_stepsize, gamma=5)
        if config.warmup:
            print("-----------using Warmup methor -----------")
            scheduler = WarmupMultiStepLR(optimizer, milestones = config.stepsize, gamma=config.gamma, warmup_factor = 0.01,                                                        warmup_iters = 10, warmup_method='linear')
        else:
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=config.stepsize, gamma=config.gamma)
        config.eval_step = 0

    if config.fixbase_epoch > 0:
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            optimizer_tmp = init_optim(config.optim, model.classifier.parameters(), config.fixbase_lr, config.weight_decay)
        else:
            print("Warn: model has no attribute 'classifier' and fixbase_epoch is reset to 0")
            config.fixbase_epoch = 0

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

    start_time = time.time()
    train_time = 0
    best_rank1 = -np.inf
    best_epoch = 0
    print("==> Start training")

    if config.fixbase_epoch > 0:
        print("Train classifier for {} epochs while keeping base network frozen".format(config.fixbase_epoch))

        for epoch in range(config.fixbase_epoch):
            start_train_time = time.time()
            train(epoch, model, criterion, optimizer_tmp, trainloader, use_gpu, freeze_bn=True)
            train_time += round(time.time() - start_train_time)

        del optimizer_tmp
        print("Now open all layers for training")

    for epoch in range(config.start_epoch, config.max_epoch):
        start_train_time = time.time()
        train(epoch, model, criterion, optimizer, trainloader, use_gpu, summary_writer, config)
        train_time += round(time.time() - start_train_time)
        
        scheduler.step()
        print('lr: %.7f'%scheduler.get_lr()[0])
        
        if (epoch + 1) > config.start_eval and config.eval_step > 0 and (epoch + 1) % config.eval_step == 0 or (epoch + 1) == config.max_epoch:
            print("==> Test")
            rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1
            
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            save_checkpoint({
                'state_dict': state_dict,
                'rank1': rank1,
                'epoch': epoch,
            }, is_best, osp.join(config.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))

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

def train(epoch, model, criterion, optimizer, trainloader, use_gpu, summary_writer, config, freeze_bn=False):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    accuracy = AverageMeter()

    model.train()

    if freeze_bn or config.freeze_bn:
        model.apply(set_bn_to_eval)

    end = time.time()
    for batch_idx, (imgs, pids, _) in enumerate(trainloader):
        data_time.update(time.time() - end)
        if use_gpu:
            pids = pids.cuda()
            if isinstance(imgs, list):
                imgs = (imgs[0].cuda(), imgs[1].cuda())
            else:
                imgs = imgs.cuda()
        
        if 'lg' in config.arch:
            outputs = model(imgs, pids)
        elif 'pg'==config.arch[:2] and config.mask_num > 0:
            outputs = model(imgs[0], imgs[1])
        else:
            outputs= model(imgs)
    #    import IPython
    #    IPython.embed()
        if isinstance(outputs, tuple):
            if config.arch=='resnet50v3ms':
                loss = MGSupervision(criterion, outputs, pids)
            else:
                loss = DeepSupervision(criterion, outputs, pids)
        else:
            loss = criterion(outputs, pids)
       
        if config.arch=='resnet50v3ms' and isinstance(outputs, tuple):
            preds = class_preds(outputs[0])
        else:
            preds = class_preds(outputs)
        corrects = float(torch.sum(preds == pids.data))
        acc = corrects / pids.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        losses.update(loss.item(), pids.size(0))
        accuracy.update(acc, pids.size(0))

        # tensorboard
        global_step = epoch * len(trainloader) + batch_idx
        summary_writer.add_scalar('loss', loss.item(), global_step)
        summary_writer.add_scalar('accuracy', acc, global_step)
        summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)

        if (batch_idx + 1) % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                   epoch + 1, batch_idx + 1, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc=accuracy))
        
        end = time.time()


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


def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    qxf, qx_pids, qx_camids = extract_feature(queryloader, model, use_gpu, flip=False)
    qf, q_pids, q_camids = extract_feature(queryloader, model, use_gpu, flip=True)
    gf, g_pids, g_camids = extract_feature(galleryloader, model, use_gpu, flip=True)
        

    result = {'gallery_f':gf.numpy(),'gallery_label':g_pids.tolist(),'gallery_cam':g_camids.tolist(),'query_f':qf.numpy(),'query_label':q_pids,'query_cam':q_camids}
    scipy.io.savemat(osp.join(config.save_dir, 'pytorch_result.mat'), result)

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r-1]))
    print("------------------")

    if return_distmat:
        return distmat
    return cmc[0]


if __name__ == '__main__':
    main()
