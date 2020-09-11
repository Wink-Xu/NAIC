from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py

class NAIC_2020(object):
    """
    NAIC_2020
    """
    dataset_dir = 'NAIC_2020'

    def __init__(self, root='../../../data', verbose=True, **kwargs):
        super(NAIC_2020, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.gallery_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'train')
        self.list_train_path = osp.join(self.dataset_dir, 'train', 'list_train_img_all_ratio.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'train', 'list_query_img_ratio.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'train', 'list_gallery_img_ratio.txt')

        self._check_before_run()
        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, self.list_train_path, relabel = 1)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, self.list_query_path)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, self.list_gallery_path)

        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        if verbose:
            print("=> NAIC_2020 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
#        if not osp.exists(self.test_dir):
#            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path, relabel = 0):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()

        temp = set()
        for line in lines:
            pid = line.split(' ')[-1]
            temp.add(pid)
        pid2label = {pid:label for label, pid in enumerate(temp)}
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            if relabel: 
                pid = pid2label[pid]
            
            #img_path = osp.join(dir_path, img_path)

            camid = 1 #int(img_path.split('_')[2])
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)
        num_imgs = len(dataset)
        num_pids = len(pid_container)
       # check if pid starts from 0 and increments with 1
        # for idx, pid in enumerate(pid_container):
        #     assert idx == pid, "See code comment for explanation "+list_path+' %d'%pid
        return dataset, num_pids, num_imgs
