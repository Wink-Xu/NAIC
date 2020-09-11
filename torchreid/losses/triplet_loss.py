# encoding: utf-8
"""
@author:  liaoxingyu
@contact: xyliao1993@qq.com 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch import nn
import torch.nn.functional as F


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


def ranking_mining(dist_mat, labels, return_inds=False):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # sort, ascend
    sorted_dist, sorted_ind = dist_mat.sort(dim=1) 
    
    # choose medium hard pairs
    a_ind_lst = []
    p_ind_lst = []
    n_ind_lst = []
    ap_lst = []
    an_lst = []
    for i in range(sorted_ind.size(0)):
        p_ind, n_ind = find_medium_hard(sorted_ind[i], is_pos[i]) 
        p_ind_lst.append(p_ind)
        n_ind_lst.append(n_ind)
        a_ind_lst.append(p_ind.new([i]*p_ind.size(0)))
        ap_lst.append(dist_mat[i][p_ind])
        an_lst.append(dist_mat[i][n_ind])
    dist_ap = torch.cat(ap_lst, 0)
    dist_an = torch.cat(an_lst, 0)

    return dist_ap, dist_an

def find_medium_hard(d_sorted_ind, d_pos):
    sorted_pos = d_pos[d_sorted_ind]
    pp = [0]
    pn = []
    pos = []
    neg = []
    for i in range(sorted_pos.size(0)):
        if sorted_pos[i] > 0:
            pp.append(i)
            if pp[-1] - pp[-2] > 1:
                pos.append(pp[-1])
                neg.append(pp[-2]+1)
        else:
            pn.append(i)
    if len(pos) > 0:
        p_ind = d_sorted_ind[pos]
        n_ind = d_sorted_ind[neg]
    else:
        p_ind = d_sorted_ind[[pp[-1]]]
        n_ind = d_sorted_ind[[pn[0]]]
    return p_ind, n_ind
    

class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False, mining='hard'): #TRY
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        if mining=='ranking':
            dist_ap, dist_an = ranking_mining(dist_mat, labels)
        elif mining=='hard':
            dist_ap, dist_an = hard_example_mining(dist_mat, labels)
        else:
            raise NameError
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss


class WeightedTripletLoss(object):
    """Related Weighted Triplet Loss theory can be found in paper
    'Attention Network Robustification for Person ReID'."""

    def __init__(self, margin=None):
        self.margin = margin

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)

        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an, final_wp, final_wn = soft_example_mining(dist_mat, labels)
        y = final_wn.new().resize_as_(final_wn).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(final_wn - final_wp, y)
        return loss

def soft_example_mining(dist_mat, labels):
    eps = 1e-12
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # shape [N]
    dist_ap = dist_mat[is_pos].contiguous().view(N, -1)
    dist_an = dist_mat[is_neg].contiguous().view(N, -1)

    exp_dist_ap = torch.exp(dist_ap)
    exp_dist_an = torch.exp(-dist_an)

    wp = exp_dist_ap / (exp_dist_ap.sum(1, keepdim=True) + eps)
    wn = exp_dist_an / (exp_dist_an.sum(1, keepdim=True) + eps)

    final_wp = (wp*dist_ap).sum(1)
    final_wn = (wn*dist_an) .sum(1)

    return dist_ap, dist_an, final_wp, final_wn