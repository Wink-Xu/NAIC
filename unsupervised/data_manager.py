from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import os.path as osp


class DataManager(object):
    def __init__(self, dataset_dir, verbose=True, **kwargs):
        super(DataManager, self).__init__()
        self.dataset_dir = dataset_dir
        self._check_before_run()

        dataset, num_pids, num_imgs = self._process_dir(self.dataset_dir, relabel=False)

        if verbose:
            print("=> %s loaded"%dataset_dir)
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  dataset    | {:5d} | {:8d}".format(num_pids, num_imgs))
            print("  ------------------------------")

        self.dataset = dataset
        self.num_pids = num_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _walk_imgs(self):
        img_paths = []
        for dirname, dirs, files in os.walk(self.dataset_dir):
            for fi in files:
                if fi.endswith('.png'):
                    img_paths.append(osp.join(dirname, fi))
        return img_paths

    def _process_dir(self, dir_path, relabel=False):
        img_paths = self._walk_imgs()
        pattern = re.compile(r'([-\d]+)_c(\d)')

        '''
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 20000  # pid == 0 means background
            assert 1 <= camid <= 100, 'camid is %d here'%(camid)
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        num_pids = len(pid_container)
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
        '''

        dataset = []
        for img_path in img_paths:
            dataset.append((img_path, 0, 0))
        num_pids = 1
        num_imgs = len(dataset)
        return dataset, num_pids, num_imgs
