from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
import torchvision
import numpy as np
import copy
from .seresnet import SEResNet50


__all__ = ['HumpMGSEResNet50V3a']


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


# |--Linear--|--bn--|--relu--|--Linear--|                                                             
class ClassBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=True, relu=True, hidden_dim=512): 
        super(ClassBlock, self).__init__()                                                            

        bottleneck = []
        bottleneck += [nn.Linear(input_dim, hidden_dim)]
        bottleneck += [nn.BatchNorm1d(hidden_dim)]
        if relu:
            bottleneck += [nn.LeakyReLU(0.1)]
        if dropout:
            bottleneck += [nn.Dropout(p=0.5)]
        self.bottleneck = nn.Sequential(*bottleneck)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat = self.bottleneck(x)
        cls_score = self.classifier(feat)                                                                        
        return cls_score, x 


################################################ MGNetV3a ##################################################
# for resnet50 pool_channels = 2048  
class MGBranchV3a(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, pool_channels=2048, embedding_dim=256):
        super(MGBranchV3a, self).__init__()

        self.parts_num = parts_num
        if self.parts_num <= 1:
            self.parts_num = 0
        # private convolution layers
        if backbone=='resnet50':
            res50 = torchvision.models.resnet50(pretrained=True)
            private_layer3 = res50.layer3
            private_layer4 = res50.layer4
            private_conv = nn.Sequential(private_layer3, private_layer4)
        elif backbone=='seresnet50':
            seres50 = SEResNet50()
            private_layer3 = seres50.base[3]
            private_layer4 = seres50.base[4]
            private_conv = nn.Sequential(private_layer3, private_layer4)
        else:
            pass
        # reset private convolution layers
        self.private_conv = private_conv
        if self.parts_num > 0:
            if backbone=='resnet50':
                self.private_conv[1][0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.private_conv[1][0].conv1.stride = (1, 1)
            self.private_conv[1][0].downsample[0].stride = (1, 1)
        if self.parts_num > 2:
        #if self.parts_num > 3:  #TRY
            if backbone=='resnet50':
                self.private_conv[0][0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.private_conv[0][0].conv1.stride = (1, 1)
            self.private_conv[0][0].downsample[0].stride = (1, 1)
        # global embedding
        self.embedding = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        # local embedding
        if self.parts_num > 0:
            self.adaptive_pool = nn.AdaptiveMaxPool2d((1, self.parts_num))
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.parts_num):
                local_emb = nn.Sequential(
                    nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(embedding_dim),
                    #nn.ReLU(inplace=True)
                )
                classifier = nn.Linear(embedding_dim, num_classes)
                local_emb.apply(weights_init_kaiming)
                classifier.apply(weights_init_classifier)
                self.local_emb_name[i] = 'local_emb%d'%(i+1)
                self.local_cls_name[i] = 'local_cls%d'%(i+1)
                setattr(self, self.local_emb_name[i], local_emb)
                setattr(self, self.local_cls_name[i], classifier)


    def forward(self, x):
        x = self.private_conv(x)
        global_feat = self.embedding(x)
        global_feat = global_feat.view(global_feat.size(0), -1)
        if self.training:
            global_score = self.classifier(global_feat)

        local_feat_group = []
        local_score_group = []
        if self.parts_num > 0:
            l_pool = self.adaptive_pool(x) 
            for i in range(self.parts_num):
                part = l_pool[:,:,:,i].unsqueeze(2)
                extr = getattr(self, self.local_emb_name[i])                                               
                feat = extr(part)
                feat = feat.view(feat.size(0), -1)
                local_feat_group.append(feat)
                if self.training:
                    cls = getattr(self, self.local_cls_name[i])                                               
                    score = cls(feat)
                    local_score_group.append(score)                                                 

        if self.training:
            return tuple([global_score] + local_score_group), tuple([global_feat])
        else:
            return tuple([global_feat] + local_feat_group)

        
class HumpMGSEResNet50V3a(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(HumpMGSEResNet50V3a, self).__init__()
        self.loss = loss

        seres50 = SEResNet50()
        self.common = nn.Sequential(seres50.base[0], seres50.base[1], seres50.base[2])
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], MGBranchV3a(num_classes, backbone='seresnet50', parts_num=num))

    def forward(self, x):
        x = self.common(x)
        y = [] 
        for i, num in enumerate(self.branch_stripes):
            branch = getattr(self, self.branch_name[i])
            y.append(branch(x))

        if self.training:
            scores = ()
            feats = ()
            for f in y:
                scores += f[0]
                feats += f[1]
            if self.loss == {'xent'}:
                return scores
            elif self.loss == {'xent', 'htri'}:
                return scores, feats 
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            feats = ()
            for f in y:
                feats += f
            feature = torch.cat(feats, 1)
            return feature



if __name__=='__main__':
    def count_num_param(model):
        num_param = sum(p.numel() for p in model.parameters()) / 1e+06
        return num_param

    net = HumpMGSEResNet50V3a(751, loss={'xent','htri'}, branch_stripes=[0,2,3])
    net.train(True)
    x = torch.Tensor(2,3,128,384)
    y = net(x)
    print(count_num_param(net))
