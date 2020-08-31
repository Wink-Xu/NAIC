from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
import torchvision
import numpy as np
import copy
from .resnet import weights_init_kaiming, weights_init_classifier
from .seresnet import SEResNet50
#import ipdb as pdb


__all__ = ['PGSEResNet50', 'PGResNet50']



# for resnet50 pool_channels = 2048  
class PGBranch(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, pool_channels=2048, embedding_dim=256):
        super(PGBranch, self).__init__()

        self.parts_num = parts_num
        if self.parts_num <= 1:
            self.parts_num = 0
        # private convolution layers
        if backbone=='resnet50':
            res50 = torchvision.models.resnet50(pretrained=True)
            self.layer3 = res50.layer3
            self.layer4 = res50.layer4
        elif backbone=='seresnet50':
            seres50 = SEResNet50()
            self.layer3 = seres50.base[3]
            self.layer4 = seres50.base[4]
        else:
            pass
        # reset private convolution layers
        if self.parts_num > 0:
            self.upsample = nn.Upsample(scale_factor=1.0/16, mode='nearest')
            if backbone=='resnet50':
                self.layer4[0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.layer4[0].conv1.stride = (1, 1)
            self.layer4[0].downsample[0].stride = (1, 1)
        #if self.parts_num > 2: # TRY
        if self.parts_num > 10:
            self.upsample = nn.Upsample(scale_factor=1.0/8, mode='nearest')
            if backbone=='resnet50':
                self.layer3[0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.layer3[0].conv1.stride = (1, 1)
            self.layer3[0].downsample[0].stride = (1, 1)
        # global embedding
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        # local embedding
        if self.parts_num > 0:
            self.adaptive_pool = nn.AdaptiveMaxPool2d((self.parts_num, 1))
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.parts_num):
                local_emb = nn.Sequential(
                    nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(embedding_dim),
                )
                classifier = nn.Linear(embedding_dim, num_classes)
                local_emb.apply(weights_init_kaiming)
                classifier.apply(weights_init_classifier)
                self.local_emb_name[i] = 'local_emb%d'%(i+1)
                self.local_cls_name[i] = 'local_cls%d'%(i+1)
                setattr(self, self.local_emb_name[i], local_emb)
                setattr(self, self.local_cls_name[i], classifier)


    def forward(self, x, mask=None):
        x = self.layer3(x)
        x = self.layer4(x)
        if mask is None:
            global_feat = self.global_pool(x)
        else:
            assert mask.size(1) == self.parts_num+1, 'mask %d, parts %d'%(mask.size(1), self.parts_num)
            mask = self.upsample(mask)
            ix = x * mask[:,0].unsqueeze(1)
            global_feat = self.global_pool(ix)
        global_feat = self.embedding(global_feat)
        global_feat = global_feat.view(global_feat.size(0), -1)
        if self.training:
            global_score = self.classifier(global_feat)

        local_feat_group = []
        local_score_group = []
        if self.parts_num > 0:
            if mask is None:
                l_pool = self.adaptive_pool(x) 
            for i in range(self.parts_num):
                if mask is None:
                    part = l_pool[:,:,i].unsqueeze(2)
                else:
                    ipart = x * mask[:,i+1].unsqueeze(1)
                    part = self.global_pool(ipart)
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

        
class PGResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(PGResNet50, self).__init__()
        self.loss = loss
        self.branch_stripes = branch_stripes

        res50 = torchvision.models.resnet50(pretrained=True)
        self.layer0 = nn.Sequential(res50.conv1, res50.bn1, res50.relu, res50.maxpool)
        self.layer1 = res50.layer1
        self.layer2 = res50.layer2
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], PGBranch(num_classes, parts_num=num))

    def forward(self, x, mask=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        y = [] 
        m_ind = 0
        for i, num in enumerate(self.branch_stripes):
            branch = getattr(self, self.branch_name[i])
            if mask is None or i==0:
                imask = None
            else:
                imask = mask[:, m_ind:m_ind+num+1]
                m_ind += (num + 1)
            y.append(branch(x, mask=imask))

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


class PGSEResNet50(nn.Module):
    #def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs): #TRY
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,6], **kwargs):
        super(PGSEResNet50, self).__init__()
        self.loss = loss
        self.branch_stripes = branch_stripes

        seres50 = SEResNet50()
        self.layer0 = seres50.base[0]
        self.layer1 = seres50.base[1]
        self.layer2 = seres50.base[2]
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], PGBranch(num_classes, backbone='seresnet50', parts_num=num))

    def forward(self, x, mask=None):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        y = [] 
        m_ind = 0
        for i, num in enumerate(self.branch_stripes):
            branch = getattr(self, self.branch_name[i])
            if mask is None or i==0:
                imask = None
            else:
                imask = mask[:, m_ind:m_ind+num+1]
                m_ind += (num + 1)
            y.append(branch(x, mask=imask))

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


from IPython import embed
if __name__=='__main__':
    def count_num_param(model):
        num_param = sum(p.numel() for p in model.parameters()) / 1e+06
        #if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            # we ignore the classifier because it is unused at test time
            #num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
        return num_param

    net = PGSEResNet50(751, loss={'xent','htri'}, branch_stripes=[0,2,3])
    net.train(False)
    print(count_num_param(net))
    net.train(True)
    x = torch.Tensor(2,3,384,128)
    mask = torch.Tensor(2,7,384,128)
    y = net(x, mask)
    embed()
