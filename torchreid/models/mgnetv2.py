from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np


__all__ = ['MGResNet50V2']


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


# for resnet50 pool_channels = 2048  
class MGBranch(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, pool_channels=2048, embedding_dim=512):
        super(MGBranch, self).__init__()

        self.parts_num = parts_num
        if self.parts_num <= 1:
            self.parts_num = 0
        # private convolution layers
        if backbone=='resnet50':
            res50 = torchvision.models.resnet50(pretrained=True)
            private_layer3 = nn.Sequential(*list(res50.layer3.children())[1:])
            private_layer4 = res50.layer4
            private_conv = nn.Sequential(private_layer3, private_layer4)
        else:
            pass
        self.private_conv = private_conv
        if self.parts_num > 0:
            self.private_conv[1][0].conv2.stride = (1, 1)
            self.private_conv[1][0].downsample[0].stride = (1, 1)
        # global embedding
        self.embedding = nn.Sequential(
            nn.Linear(pool_channels, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5)
        )
        self.embedding.apply(weights_init_kaiming)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.classifier.apply(weights_init_classifier)
        # local embedding
        if self.parts_num > 0:
            self.local_pool = nn.AdaptiveAvgPool2d((self.parts_num, 1))
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.parts_num):
                local_emb = nn.Sequential(
                    nn.Linear(pool_channels, embedding_dim),
                    nn.BatchNorm1d(embedding_dim),
                    nn.LeakyReLU(0.1),
                    nn.Dropout(p=0.5)
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
        g_pool = F.avg_pool2d(x, x.size()[2:])
        global_feat = g_pool.view(g_pool.size(0), -1)
        if self.training:
            y = self.embedding(global_feat)
            global_score = self.classifier(y)

        local_feat_group = []
        local_score_group = []
        if self.parts_num > 0:
            l_pool = self.local_pool(x)
            for i in range(self.parts_num):
                feat = torch.squeeze(l_pool[:,:,i])
                local_feat_group.append(feat)
                if self.training:
                    emb = getattr(self, self.local_emb_name[i])                                               
                    cls = getattr(self, self.local_cls_name[i])                                               
                    embedding = emb(feat)
                    score = cls(embedding)
                    local_score_group.append(score)                                                 

        if self.training:
            return tuple([global_score] + local_score_group), tuple([global_feat])
        else:
            return tuple([global_feat] + local_feat_group)

        
class MGResNet50V2(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGResNet50V2, self).__init__()
        self.loss = loss

        res50 = torchvision.models.resnet50(pretrained=True)
        self.common = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2, res50.layer3[0]
        )
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], MGBranch(num_classes, parts_num=num))

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
        #if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            # we ignore the classifier because it is unused at test time
            #num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
        return num_param

    #net = torchvision.models.resnet50(pretrained=True)
    #base = nn.Sequential(*list(net.children())[:-2])
    net = MGResNet50V2(751, loss={'xent','htri'}, branch_stripes=[0])
    base = net.common
    print(count_num_param(net))
    print(count_num_param(base))
    net.train(True)
    x = torch.Tensor(2,3,384,128)
    y = net(x)
    from IPython import embed
    embed()
