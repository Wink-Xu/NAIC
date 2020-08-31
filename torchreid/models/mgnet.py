from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
import torchvision
import numpy as np


__all__ = ['MGResNet50', 'PCResnet50']


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
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, pool_channels=2048, embedding_dim=256):
        super(MGBranch, self).__init__()

        self.parts_num = parts_num
        if self.parts_num <= 1:
            self.parts_num = 0
        # private convolution layers
        if backbone=='resnet50':
            res50 = torchvision.models.resnet50(pretrained=True)
            private_layer3 = nn.Sequential(*list(res50.layer3.children())[1:])
            private_layer4 = nn.Sequential(res50.layer4[0])
            private_conv = nn.Sequential(private_layer3, private_layer4)
        else:
            pass
        self.private_conv = private_conv
        if self.parts_num > 0:
            self.private_conv[1][0].conv2.stride = (1, 1)
            self.private_conv[1][0].downsample[0].stride = (1, 1)
        # global embedding
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.embedding = nn.Sequential(
            nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        self.embedding.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_channels, num_classes)
        self.classifier.apply(weights_init_classifier)
        # local embedding
        if self.parts_num > 0:
            self.local_pool = nn.AdaptiveMaxPool2d((self.parts_num, 1))
            self.local_emb = nn.Sequential(
                nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embedding_dim),
                nn.ReLU(inplace=True)
            )
            self.local_emb.apply(weights_init_kaiming)
            self.local_cls_name = {}
            for i in range(self.parts_num):
                self.local_cls_name[i] = 'local_cls%d'%(i+1)
                classifier = nn.Linear(embedding_dim, num_classes)
                classifier.apply(weights_init_classifier)
                setattr(self, self.local_cls_name[i], classifier)


    def forward(self, x):
        x = self.private_conv(x)
        g_pool = self.global_pool(x)
        global_feat = self.embedding(g_pool)
        global_feat = torch.squeeze(global_feat)
        if self.training:
            g_pool = torch.squeeze(g_pool)
            global_score = self.classifier(g_pool)

        local_feat_group = []
        local_score_group = []
        if self.parts_num > 0:
            l_pool = self.local_pool(x)
            local_feat = self.local_emb(l_pool)
            for i in range(self.parts_num):
                feat = torch.squeeze(local_feat[:,:,i])
                local_feat_group.append(feat)
                if self.training:
                    cls = getattr(self, self.local_cls_name[i])                                               
                    score = cls(feat)
                    local_score_group.append(score)                                                 

        if self.training:
            return tuple([global_score] + local_score_group), tuple([global_feat])
        else:
            return tuple([global_feat] + local_feat_group)

        
class MGResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGResNet50, self).__init__()
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



# Part Model proposed in Yifan Sun etal. (2018)
class PCResnet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(PCResnet50, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = torchvision.models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, num_classes, True, False, 256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)

        if not self.training:
            ff = x.view(x.size(0),x.size(1),x.size(2))
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
            return ff

        else:
            x = self.dropout(x)
            part = {}
            predict = {}
            # get six part feature batchsize*2048*6
            for i in range(self.part):
                part[i] = torch.squeeze(x[:,:,i])
                name = 'classifier'+str(i)
                c = getattr(self,name)
                predict[i],_ = c(part[i])

            # sum prediction
            y = []
            for i in range(self.part):
                y.append(predict[i])
            return tuple(y)



if __name__=='__main__':
    def count_num_param(model):
        num_param = sum(p.numel() for p in model.parameters()) / 1e+06
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            # we ignore the classifier because it is unused at test time
            num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
        return num_param

    net = torchvision.models.resnet50(pretrained=True)
    base = nn.Sequential(*list(net.children())[:-2])
    #net = MGResNet50(751, loss={'xent','htri'}, branch_stripes=[0])
    #base = net.common
    #net = PCResnet50(200)
    print(count_num_param(net))
    print(count_num_param(base))
    net.train(True)
    x = torch.Tensor(2,3,384,128)
    y = net(x)
    from IPython import embed
    embed()
