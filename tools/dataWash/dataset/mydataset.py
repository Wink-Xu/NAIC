from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import os
import json_tricks as json
import cv2

import numpy as np
from scipy.io import loadmat, savemat

import json_tricks as json

from .JointsDataset2 import JointsDataset2

logger = logging.getLogger(__name__)

class mydataset(JointsDataset2):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        # self.coco = COCO(self._get_ann_file_keypoint())

        self.num_joints = 17
        self.flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]


        self.db = self._get_db()

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):


        image_dir = os.path.join(self.root,self.image_set)
        image_path_list = os.listdir(image_dir)


        gt_db = []
        for image_name in image_path_list:
            temp_path = os.path.join(image_dir,image_name)
            data_numpy = cv2.imread(
                temp_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            h = data_numpy.shape[0]
            w = data_numpy.shape[1]
            c,s = self.get_c_s(w=w,h=h)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            gt_db.append({
                'image':temp_path,
                'center': c,
                'scale' : s,
                'score' : 1,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': image_name,
                'imgnum': 0,
            })

        return gt_db

    def get_c_s(self, h, w):
        res_h = 384
        res_w = 288
        center = np.zeros((2), dtype=np.float32)
        center[0] = w * 0.5
        center[1] = h * 0.5
        '''
        scale = np.array([min(res_w / w , res_w , h),min(res_h / w , res_h , h)])


        if center[0] != -1:
            scale = scale * 1.25


        '''
        # scale = np.array([  h / 200, w /
        temp = max(h, w)
        temp = temp * 0.75
        scale = np.array([temp / 200, temp / 200])
        # scale = np.array([0.48, 0.48])
        return center, scale

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):

        output_dir = os.path.join(output_dir, cfg.DATASET.TEST_SET + '_json')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'image': img_path[idx]
            })
        # json_data = []
        for ii, kk in enumerate(_kpts):
            img_path = kk['image']
            img_name = img_path.split('/')[-1]
            img_name_list = img_name.split('.')

            if len(img_name_list) <=2:
                jsom_name = img_name_list[0]
            else :
                jsom_name = img_name_list[0] + '.' +img_name_list[1]

            jsom_name = jsom_name + '.json'
            json_path = os.path.join(output_dir,jsom_name)
            json_data_temp = [{
                'keypoints': kk['keypoints'],
            }]
            with open(json_path, 'w') as f:
                json.dump(json_data_temp, f, sort_keys=True, indent=4)

        return {'Null': 0}, 0
































    def _compute_mean(self):
        meanstd_file = ''
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else :
            mean = torch.zeros(3)
            std = torch.zeros(3)

            for ind, im_name in []:
                img = load_image(im_name)

                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.dic['image_list'])
            std /= len(self.dic['image_list'])
            meanstd = {
                'mean': mean,
                'std': std,
            }
            torch.save(meanstd, meanstd_file)




