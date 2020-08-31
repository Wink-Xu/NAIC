from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision


__all__ = ['DenseNet121']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        nn.init.constant_(m.bias, 0.0)


class DenseNet121(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(DenseNet121, self).__init__()
        self.loss = loss
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        #self.classifier = nn.Linear(1024, num_classes)
        self.feat_dim = 1024

        # add bottleneck layer before classifier     
        self.bottleneck = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5)
        )
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.bottleneck(f)
        y = self.classifier(y)
        
        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))



if __name__=='__main__':
    def count_num_param(model):
        num_param = sum(p.numel() for p in model.parameters()) / 1e+06
        return num_param

    net = DenseNet121(1000, loss={'xent','htri'})
    print(count_num_param(net))
    net.train(True)
    x = torch.Tensor(2,3,256,128)
    y = net(x)
