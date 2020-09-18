from __future__ import absolute_import
from __future__ import division

#from torchvision.transforms import *
import torch
import collections
from torchvision.transforms import Normalize
from torchvision.transforms import functional as F

from PIL import Image
import math
import random
import numpy as np
import Augmentor
import ipdb as pdb
from collections import deque

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img
        
   
        for attempt in range(100):
 
           # pdb.set_trace()
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img



class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target height.
    - width (int): target width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def _resize(self, img, labels, new_w, new_h):
        img = img.resize((new_w, new_h), self.interpolation)
        for i in range(len(labels)):
            labels[i] = labels[i].resize((new_w, new_h), Image.NEAREST)
        return img, labels

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            if isinstance(img, list):
                img, labels = self._resize(img[0], img[1:], self.width, self.height)
                return [img] + labels
            else:
                return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        if isinstance(img, list):
            resized_img, labels = self._resize(img[0], img[1:], new_width, new_height)
            croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
            for i in range(len(labels)):
                labels[i] = labels[i].crop((x1, y1, x1 + self.width, y1 + self.height))
            return [cropped_img] + labels
        else:
            resized_img = img.resize((new_width, new_height), self.interpolation)
            croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
            return croped_img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class ToTensor(object):
    """Convert several ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts several PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, pic):
        if isinstance(pic, list):
            tensor = []
            for p in pic:
                tensor.append(F.to_tensor(p))
            return torch.cat(tensor, 0)
        else:
            return F.to_tensor(pic)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img):
        if random.random() < 0.5:
            if isinstance(img, list):
                for i in range(len(img)):
                    img[i] = F.hflip(img[i])
                return img 
            else:
                return F.hflip(img)
        return img


class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        if isinstance(img, list):
            for i in range(len(img)):
                img[i] = F.resize(img[i], self.size, self.interpolation)
            return img
        else:
            return F.resize(img, self.size, self.interpolation)


def RandomAugment(IP=False, Graph=False, Erase=False):
    if IP==False and Graph==False and Erase==False:
        return None
    p = Augmentor.Pipeline()

    if IP:
        p.random_color(0.5, min_factor=0.4, max_factor=1.6)
        p.random_brightness(0.5, min_factor=0.4, max_factor=1.6)
        p.random_contrast(0.5, min_factor=0.4, max_factor=1.2)

    if Graph:
        p.rotate(probability=0.7, max_left_rotation=7, max_right_rotation=7)
        p.zoom(probability=0.5, min_factor=0.8, max_factor=1.2)
        #p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=4)
        p.skew_left_right(probability=0.5, magnitude=0.15)

    if Erase:
        p.random_erasing(1.0,rectangle_area=0.5)

    return p



class RandomPatch(object):
    """Random patch data augmentation.
    There is a patch pool that stores randomly extracted pathces from person images.
    For each input image, RandomPatch
        1) extracts a random patch and stores the patch in the patch pool;
        2) randomly selects a patch from the patch pool and pastes it on the
           input (at random position) to simulate occlusion.
    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(self, prob_happen=0.5, pool_capacity=50000, min_sample_size=100,
                 patch_min_area=0.01, patch_max_area=0.5, patch_min_ratio=0.1,
                 prob_rotate=0.5, prob_flip_leftright=0.5,
                 ):
        self.prob_happen = prob_happen

        self.patch_min_area = patch_min_area
        self.patch_max_area = patch_max_area
        self.patch_min_ratio = patch_min_ratio

        self.prob_rotate = prob_rotate
        self.prob_flip_leftright = prob_flip_leftright

        self.patchpool = deque(maxlen=pool_capacity)
        self.min_sample_size = min_sample_size

    def generate_wh(self, W, H):
        area = W * H
        for attempt in range(100):
            target_area = random.uniform(self.patch_min_area, self.patch_max_area) * area
            aspect_ratio = random.uniform(self.patch_min_ratio, 1. / self.patch_min_ratio)
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < W and h < H:
                return w, h
        return None, None

    def transform_patch(self, patch):
        if random.uniform(0, 1) > self.prob_flip_leftright:
            patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
        if random.uniform(0, 1) > self.prob_rotate:
            patch = patch.rotate(random.randint(-10, 10))
        return patch

    def __call__(self, img):
        W, H = img.size  # original image size

        # collect new patch
        w, h = self.generate_wh(W, H)
        if w is not None and h is not None:
            x1 = random.randint(0, W - w)
            y1 = random.randint(0, H - h)
            new_patch = img.crop((x1, y1, x1 + w, y1 + h))
            self.patchpool.append(new_patch)

        if len(self.patchpool) < self.min_sample_size:
            return img

        if random.uniform(0, 1) > self.prob_happen:
            return img

        # paste a randomly selected patch on a random position
        patch = random.sample(self.patchpool, 1)[0]
        patchW, patchH = patch.size
        x1 = random.randint(0, W - patchW)
        y1 = random.randint(0, H - patchH)
        patch = self.transform_patch(patch)
        img.paste(patch, (x1, y1))

        return img