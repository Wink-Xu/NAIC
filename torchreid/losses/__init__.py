from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .cross_entropy_loss import CrossEntropyLabelSmooth, CrossEntropyWithMaxEntropy
#from .hard_mine_triplet_loss import TripletLoss
from .triplet_loss import TripletLoss
from .triplet_loss import WeightedTripletLoss
from .center_loss import CenterLoss
from .ring_loss import RingLoss
from .circle_loss import LabelCircleLoss
from .CrossEntropyFastReID import CrossEntropyLossFastReID 
from .oim_loss import OIMLoss
import pdb
import torch



def CircleSupervision(criterion, xs, y):
    loss = 0.
    temp_x = xs[0]
    for x in xs[1:]:
        temp_x = torch.cat((temp_x, x), 1)
    loss = criterion(temp_x, y)
#    pdb.set_trace()
    return loss

def DeepSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss


def MGSupervision(criterion, xs, y):
    """
    Args:
    - criterion: loss function
    - xs: tuple of inputs
    - y: ground truth
    """
    loss = 0.
    for x in xs:
        #ipdb.set_trace()
        if len(x.shape) > 2:
            n, c, h, w = x.size()
            f = x.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
            t = y.unsqueeze(1).repeat(1, h*w)
            t = t.view(-1)
            loss += criterion(f, t)
        else:
            loss += criterion(x, y)
    loss /= len(xs)
    return loss
