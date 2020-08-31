from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
import numpy as np
import os.path as osp
import io

import torch
from torch.utils.data import Dataset
import json
import os.path as osp
#import ipdb as pdb


def read_image(img_path, format='RGB'):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path)
            if format=='RGB':
                img = img.convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def save_image(img_path, img):
    ''' saving image '''
    img.save(img_path)
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None, use_kpts=False, mask_num=0):
        self.dataset = dataset
        self.transform = transform
        self.use_kpts = use_kpts
        self.mask_num = mask_num
        self.file_tree_type = 0

    def __len__(self):
        return len(self.dataset)

    def get_mask_dir(self, filename):
        t_words = filename.split('/')
        assert len(t_words) >= 3, 'Invalid file name'
        hyp_dir1 = osp.join(*t_words[:-2], t_words[-2]+'_mask')
        hyp_fil1 = t_words[-1]
        hyp_dir2 = osp.join(*t_words[:-3], t_words[-3]+'_mask')
        hyp_fil2 = osp.join(*t_words[-2:])
        if self.file_tree_type == 0:
            if osp.exists(hyp_dir1):
                self.file_tree_type = 1
            elif osp.exists(hyp_dir2):
                self.file_tree_type = 2
            assert self.file_tree_type > 0, 'Can not find mask files'
        if self.file_tree_type==1:
            return osp.join(hyp_dir1, osp.splitext(hyp_fil1)[0])
        else:
            return osp.join(hyp_dir2, osp.splitext(hyp_fil2)[0])

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
  
        if self.use_kpts:
            json_path = img_path.replace('.jpg', '.json')
            key_pts = img.new([-1,-1]) 
            if osp.exists(json_path):
                with open(json_path) as jf:
                    key_pts = json.load(jf)
                    key_pts = torch.Tensor(key_pts['keypoints'])
                    key_pts[:,0] /= img.width
                    key_pts[:,1] /= img.height

        if self.mask_num > 0:
            imgs = [img]
            mask_dir = self.get_mask_dir(img_path)
            for i in range(self.mask_num):
                mask_path = osp.join(mask_dir, '%02d.jpg'%i)
                assert osp.exists(mask_path), mask_path
                mask = read_image(mask_path, format='GRAY')
                imgs.append(mask)

        if self.transform is not None:
            if self.mask_num > 0:
                #pdb.set_trace()
                imgs = self.transform(imgs)
                img = imgs[:imgs.size(0)-self.mask_num]
                masks = imgs[imgs.size(0)-self.mask_num:]
            else:
                img = self.transform(img)
        
        if self.use_kpts:
            return (img,key_pts), pid, camid
        if self.mask_num > 0:
            return (img,masks), pid, camid
        return img, pid, camid


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)
        elif self.sample == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num/self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid
