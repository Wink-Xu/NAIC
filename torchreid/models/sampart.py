from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
import torchvision
import numpy as np
import copy
from .resnet import weights_init_kaiming, weights_init_classifier
from .seresnet import SEResNet50
from .drop import SamplePart
import ipdb as pdb


__all__ = ['SPResNet50V3a', 'SPSEResNet50V3a', 'SPResNet50', 'SPResNet50V2']


#
class SPBranch(nn.Module):
    def __init__(self, num_classes, subnet, inplanes=2048, embedding_dim=256, **kwargs):
        super(SPBranch, self).__init__()
        self.part_size = kwargs.get('part_size', 0)
        self.parts_num = kwargs.get('parts_num', 0)

        self.subnet = copy.deepcopy(subnet)
        if self.part_size > 1:
            self.sampler = SamplePart(self.part_size, self.parts_num)
        self.embedding = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(inplanes, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, ind=0):
        x = self.subnet(x)
        if self.part_size > 1:
            x = self.sampler(x, ind)

        x = self.embedding(x)
        f = x.view(x.size(0), -1)

        if self.training:
            y = self.classifier(f)
            return y, f
        else:
            return f


class SPResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(SPResNet50, self).__init__()
        self.part_size = kwargs.get('part_size', (2,))
        self.parts_num = kwargs.get('parts_num', (4,)) #TRY
        #self.part_size = kwargs.get('part_size', (2,6))
        #self.parts_num = kwargs.get('parts_num', (4,12)) #TRY
        self.loss = loss

        # backbone
        res50 = torchvision.models.resnet50(pretrained=True)
        res50.layer4[0].conv2.stride = (1, 1)
        res50.layer4[0].downsample[0].stride = (1, 1)
        self.backbone = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2, res50.layer3[0]
        )
        self.backbone_s1 = nn.Sequential(res50.layer3[1:])
        self.backbone_s2 = nn.Sequential(res50.layer4[:2])

        # global subnet
        subnet = res50.layer4[2]
        self.global_subnet = SPBranch(num_classes, subnet, part_size=1)

        # sample part branches
        for i, parts_num in enumerate(self.parts_num):
            part_size = self.part_size[i]
            for p in range(parts_num):
                branch_name = 'branch%d_%d'%(part_size, p)
                setattr(self, branch_name, SPBranch(num_classes, subnet, part_size=part_size, parts_num=parts_num))

    def forward(self, x):
        x = self.backbone(x)
        x1 = self.backbone_s1(x)
        x2 = self.backbone_s2(x1)
        x = torch.cat((x, x1), 1) + x2

        y = [] 
        y.append(self.global_subnet(x))
        for i, parts_num in enumerate(self.parts_num):
            part_size = self.part_size[i]
            for p in range(parts_num):
                branch = getattr(self, 'branch%d_%d'%(part_size, p))
                y.append(branch(x, p))

        if self.training:
            scores = ()
            feats = ()
            for f in y:
                scores += (f[0],)
                feats += (f[1],)
            if self.loss == {'xent'}:
                return scores
            elif self.loss == {'xent', 'htri'}:
                return scores, feats 
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            feats = ()
            for f in y:
                feats += (f,)
            feature = torch.cat(feats, 1)
            return feature


class SPResNet50V2(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(SPResNet50V2, self).__init__()
        self.part_size = kwargs.get('part_size', (2,3))
        self.parts_num = kwargs.get('parts_num', (2,3)) #TRY
        self.loss = loss

        # backbone
        res50 = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2, res50.layer3
        )
        layer4 = res50.layer4[:2]
        layer4[0].conv2.stride = (1, 1)
        layer4[0].downsample[0].stride = (1, 1)
        self.global_crotch = nn.Sequential(layer4)
        for i in range(len(self.parts_num)):
            setattr(self, 'crotch_%d'%i, nn.Sequential(copy.deepcopy(layer4)))

        # global subnet
        subnet = res50.layer4[2]
        self.global_subnet = SPBranch(num_classes, subnet, part_size=1)

        # sample part branches
        for i, parts_num in enumerate(self.parts_num):
            part_size = self.part_size[i]
            branch_name = 'branch%d_g'%(i)
            setattr(self, branch_name, SPBranch(num_classes, subnet, part_size=1))
            for p in range(parts_num):
                branch_name = 'branch%d_%d'%(i, p)
                setattr(self, branch_name, SPBranch(num_classes, subnet, part_size=part_size, parts_num=parts_num))

    def forward(self, x):
        x = self.backbone(x)

        y = [] 
        xg = self.global_crotch(x)
        y.append(self.global_subnet(xg))
        for i, parts_num in enumerate(self.parts_num):
            part_size = self.part_size[i]
            crotch = getattr(self, 'crotch_%d'%i)
            branch = getattr(self, 'branch%d_g'%(i))
            x1 = crotch(x)
            y.append(branch(x1))
            for p in range(parts_num):
                branch = getattr(self, 'branch%d_%d'%(i, p))
                y.append(branch(x1, p))

        if self.training:
            scores = ()
            feats = ()
            for f in y:
                scores += (f[0],)
                feats += (f[1],)
            if self.loss == {'xent'}:
                return scores
            elif self.loss == {'xent', 'htri'}:
                return scores, feats 
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            feats = ()
            for f in y:
                feats += (f,)
            feature = torch.cat(feats, 1)
            return feature


# for resnet50 pool_channels = 2048  
class SPBranchV3a(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, part_size=0, pool_channels=2048, embedding_dim=256):
        super(SPBranchV3a, self).__init__()

        self.parts_num = parts_num
        self.part_size = part_size
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
        if self.part_size > 0:
            if backbone=='resnet50':
                self.private_conv[1][0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.private_conv[1][0].conv1.stride = (1, 1)
            self.private_conv[1][0].downsample[0].stride = (1, 1)
        if self.part_size > 2:
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
            self.sampart = SamplePart(self.part_size, self.parts_num)
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.parts_num):
                local_emb = nn.Sequential(
                    nn.AdaptiveMaxPool2d((1, 1)),
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


    def forward(self, x):
        x = self.private_conv(x)
        global_feat = self.embedding(x)
        global_feat = global_feat.view(global_feat.size(0), -1)
        if self.training:
            global_score = self.classifier(global_feat)

        local_feat_group = []
        local_score_group = []
        if self.parts_num > 0:
            for i in range(self.parts_num):
                part = self.sampart(x, i)                                               
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

        
class SPResNet50V3a(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], branch_num=[0,2,3], **kwargs):
        super(SPResNet50V3a, self).__init__()
        self.loss = loss

        res50 = torchvision.models.resnet50(pretrained=True)
        self.common = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2
        )
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, s in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], SPBranchV3a(num_classes, parts_num=branch_num[i], part_size=s))

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


class SPSEResNet50V3a(nn.Module):
    #def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], branch_num=[0,2,3], **kwargs):
    #def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], branch_num=[0,8,12], **kwargs): #TRY
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], branch_num=[0,4,6], **kwargs): #TRY
        super(SPSEResNet50V3a, self).__init__()
        self.loss = loss

        seres50 = SEResNet50()
        self.common = nn.Sequential(seres50.base[0], seres50.base[1], seres50.base[2])
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, s in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], SPBranchV3a(num_classes, backbone='seresnet50', parts_num=branch_num[i], part_size=s))

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
    from IPython import embed
    def count_num_param(model):
        num_param = sum(p.numel() for p in model.parameters()) / 1e+06
        #if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            # we ignore the classifier because it is unused at test time
            #num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
        return num_param

    #net = torchvision.models.resnet50(pretrained=True)
    #base = nn.Sequential(*list(net.children())[:-2])
    net = SPSEResNet50V3a(751, loss={'xent','htri'}, branch_stripes=[0,2,3])
    net.train(False)
    base = net.common
    print(count_num_param(net))
    print(count_num_param(base))
    net.train(True)
    x = torch.Tensor(2,3,384,128)
    y = net(x)
