# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:04:03 2020

@author: Wink
"""


import torch 
import torch.nn as nn
import math

class LabelCircleLoss(nn.Module):
    def __init__(self, num_classes, m=0.25, gamma=128, feature_dim=2048, use_gpu = True):
        super(LabelCircleLoss, self).__init__()
        self.margin = m
        self.gamma = gamma
        self.weight = torch.nn.Parameter(torch.randn(feature_dim, num_classes, requires_grad=True, device = "cuda"))
        self.labels = torch.tensor([x for x in range(num_classes)]).cuda()
        self.classes = num_classes
  #      self.init_weights()
        self.O_p = 1 + self.margin
        self.O_n = -self.margin
        self.Delta_p = 1 - self.margin
        self.Delta_n = self.margin
        self.loss = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#    def init_weights(self, pretrained=None):
#        self.weight.data.normal_()

    def forward(self, feat, label):
        normed_feat = torch.nn.functional.normalize(feat)
        normed_weight = torch.nn.functional.normalize(self.weight,dim=0)

        bs = label.size(0)
#        mask = label.expand(self.classes, bs).t().eq(self.labels.expand(bs,self.classes)).float() 
        y_true = torch.zeros((bs,self.classes),device="cuda").scatter_(1,label.view(-1,1),1)
        y_pred = torch.mm(normed_feat,normed_weight)
        y_pred = y_pred.clamp(-1,1)
#        sp = y_pred[mask == 1]
#        sn = y_pred[mask == 0]

        alpha_p = (self.O_p - y_pred.detach()).clamp(min=0)
        alpha_n = (y_pred.detach() - self.O_n).clamp(min=0)

        y_pred = (y_true * (alpha_p * (y_pred - self.Delta_p)) +
                    (1-y_true) * (alpha_n * (y_pred - self.Delta_n))) * self.gamma
        loss = self.loss(y_pred,label)

        return loss


if __name__ == '__main__':
    x = LabelCircleLoss(2048)
    import IPython
    IPython.embed()
