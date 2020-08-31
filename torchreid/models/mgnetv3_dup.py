from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
import torchvision
import numpy as np
import copy
import math
from .seresnet import SEResNet50


__all__ = ['MGResNet50V3', 'MGResNet50V3a', 'MGSEResNet50V3', 'MGSEResNet50V3a', 'MGSEResNet50V3b', 'MGSEResNet50V3c', 'MGSEResNet50V3d']



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock_IBN_b(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock_IBN_b, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck_IBN_b(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, IN=False):
        super(Bottleneck_IBN_b, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(planes * 4, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out


class ResNet_IBN_b(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        scale = 64
        self.inplanes = scale
        super(ResNet_IBN_b, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(scale, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0], stride=1, IN=True)
        self.layer2 = self._make_layer(block, scale*2, layers[1], stride=2, IN=True)
        self.layer3 = self._make_layer(block, scale*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale*8, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, IN=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, IN=IN))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

       # x = self.avgpool(x)
       # x = x.view(x.size(0), -1)
       # x = self.fc(x)

        return x
		
    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i[7:]].copy_(param_dict[i])



def resnet50_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN_b(Bottleneck_IBN_b, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_param('/data/xuzihao/ReID/code/resnet50_ibn_b.pth.tar')
    return model


def resnet101_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck_IBN_b, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck_IBN_b, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
class IBN(nn.Module):
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes/2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)
    
    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class Bottleneck_IBN(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, ibn=False, stride=1, downsample=None):
        super(Bottleneck_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn:
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_IBN(nn.Module):

    def __init__(self, last_stride, block, layers, num_classes=1000):
        scale = 64
        self.inplanes = scale
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, scale, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(scale)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, scale, layers[0])
        self.layer2 = self._make_layer(block, scale*2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, scale*4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, scale*8, layers[3], stride=last_stride)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(scale * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        ibn = True
        if planes == 512:
            ibn = False
        layers.append(block(self.inplanes, planes, ibn, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

    def load_param(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])


def resnet50_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_param('/data/xuzihao/ReID/code/r50_ibn_a.pth')
    return model


def resnet101_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_ibn_a(last_stride, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(last_stride, Bottleneck_IBN, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



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


################################################ MGNetV3 ###################################################
# for resnet50 pool_channels = 2048  
class MGBranchV3(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, pool_channels=2048, embedding_dim=256):
        super(MGBranchV3, self).__init__()

        self.parts_num = parts_num
        if self.parts_num <= 1:
            self.parts_num = 0
        # private convolution layers
        if backbone=='resnet50':
            res50 = torchvision.models.resnet50(pretrained=True)
            private_layer3 = res50.layer3[1:]
            private_layer4 = res50.layer4
            private_conv = nn.Sequential(private_layer3, private_layer4)
        elif backbone=='seresnet50':
            seres50 = SEResNet50()
            private_layer3 = seres50.base[3][1:]
            private_layer4 = seres50.base[4]
            private_conv = nn.Sequential(private_layer3, private_layer4)
        else:
            pass
        self.private_conv = private_conv
        #if self.parts_num > 0:
        if True:
            if backbone=='resnet50':
                self.private_conv[1][0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.private_conv[1][0].conv1.stride = (1, 1)
            self.private_conv[1][0].downsample[0].stride = (1, 1)
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
            self.adaptive_pool = nn.AdaptiveMaxPool2d((self.parts_num, 1))
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
                part = l_pool[:,:,i].unsqueeze(2)
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

        
class MGResNet50V3(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGResNet50V3, self).__init__()
        self.loss = loss

        res50 = torchvision.models.resnet50(pretrained=True)
        self.common = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2, res50.layer3[0]
        )
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], MGBranchV3(num_classes, parts_num=num))

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


class MGSEResNet50V3(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGSEResNet50V3, self).__init__()
        self.loss = loss

        seres50 = SEResNet50()
        self.common = nn.Sequential(seres50.base[0], seres50.base[1], 
            seres50.base[2], seres50.base[3][0]
        )
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], MGBranchV3(num_classes, backbone='seresnet50', parts_num=num))

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


################################################ MGNetV3a ##################################################
# for resnet50 pool_channels = 2048  
# AdaptiveMaxPool2d could not been supported by ONNX
class MGBranchV3a(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, pool_channels=2048, embedding_dim=256, onnx_en=False):
        super(MGBranchV3a, self).__init__()
        self.onnx_en = onnx_en
        self.parts_num = parts_num
        if self.parts_num <= 1:
            self.parts_num = 0
        # private convolution layers
        if backbone=='resnet50':
        #    res50 = torchvision.models.resnet50(pretrained=True)
          #  res50 = resnet50_ibn_a(1, pretrained = True)
            res50 = resnet50_ibn_b(pretrained=True)
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
            if backbone=='resnet50':
                self.private_conv[0][0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.private_conv[0][0].conv1.stride = (1, 1)
            self.private_conv[0][0].downsample[0].stride = (1, 1)
        # global embedding
        if onnx_en:
            factor_dict = {0:1, 2:2, 3:4}
            factor = factor_dict[self.parts_num]
            pool_h = int(12 * factor)
            pool_w = int(4 * factor)
            global_pool = nn.MaxPool2d((pool_h, pool_w), stride=(pool_h, pool_w))
        else:
            global_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.embedding = nn.Sequential(
            global_pool,
            nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
        )
        self.embedding.apply(weights_init_kaiming)
        if not onnx_en:
            self.classifier = nn.Linear(embedding_dim, num_classes)
            self.classifier.apply(weights_init_classifier)
        # local embedding
        if self.parts_num > 0:
            if onnx_en:
                p_pool_h = pool_h // int(self.parts_num)
                self.adaptive_pool = nn.MaxPool2d((p_pool_h, pool_w), stride=(p_pool_h, pool_w))
            else:
                self.adaptive_pool = nn.AdaptiveMaxPool2d((self.parts_num, 1))
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.parts_num):
                if onnx_en:
                    p_conv = nn.Conv2d(pool_channels, embedding_dim, kernel_size=(parts_num,1), bias=False)
                else:
                    p_conv = nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False)
                local_emb = nn.Sequential(
                    p_conv,
                    nn.BatchNorm2d(embedding_dim),
                    #nn.ReLU(inplace=True)
                )
                local_emb.apply(weights_init_kaiming)
                self.local_emb_name[i] = 'local_emb%d'%(i+1)
                setattr(self, self.local_emb_name[i], local_emb)
                if not onnx_en:
                    classifier = nn.Linear(embedding_dim, num_classes)
                    classifier.apply(weights_init_classifier)
                    self.local_cls_name[i] = 'local_cls%d'%(i+1)
                    setattr(self, self.local_cls_name[i], classifier)

    def forward(self, x):
        x = self.private_conv(x)
        global_feat = self.embedding(x)
        if self.training:
            global_feat = global_feat.view(global_feat.size(0), -1)
            global_score = self.classifier(global_feat)

        local_feat_group = []
        local_score_group = []
        if self.parts_num > 0:
            l_pool = self.adaptive_pool(x) 
            for i in range(self.parts_num):
                if self.onnx_en:
                    part = l_pool
                else:
                    part = l_pool[:,:,i].unsqueeze(2)
                extr = getattr(self, self.local_emb_name[i])                                               
                feat = extr(part)
                if self.training:
                    feat = feat.view(feat.size(0), -1)
                    cls = getattr(self, self.local_cls_name[i])                                               
                    score = cls(feat)
                    local_score_group.append(score)                                                 
                local_feat_group.append(feat)

        if self.training:
            return tuple([global_score] + local_score_group), tuple([global_feat])
        else:
            return tuple([global_feat] + local_feat_group)

        
class MGResNet50V3a(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGResNet50V3a, self).__init__()
        self.loss = loss
        self.onnx_en = False 

       # res50 = torchvision.models.resnet50(pretrained=True)
      #  res50 = resnet50_ibn_a(1, pretrained = True)
        res50 = resnet50_ibn_b(pretrained = True)
        self.common = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2
        )
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], MGBranchV3a(num_classes, parts_num=num, onnx_en=self.onnx_en))

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
            if not self.onnx_en:
                feature = feature.view(feature.size(0), -1)
            return feature


class MGSEResNet50V3a(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGSEResNet50V3a, self).__init__()
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



################################################ MGNetV3b ##################################################
# for resnet50 pool_channels = 2048  
class MGBranchV3b(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, pool_channels=2048, embedding_dim=256):
        super(MGBranchV3b, self).__init__()

        self.parts_num = parts_num
        if self.parts_num <= 1:
            self.parts_num = 0
        # private convolution layers
        if backbone=='resnet50':
            res50 = torchvision.models.resnet50(pretrained=True)
            private_layer3 = res50.layer3
            private_layer4 = res50.layer4
            if parts_num==0:
                private_conv = nn.Sequential(private_layer3, private_layer4)
            else:
                private_conv = nn.Sequential(private_layer3)
        elif backbone=='seresnet50':
            seres50 = SEResNet50()
            private_layer3 = seres50.base[3]
            private_layer4 = seres50.base[4]
            if parts_num==0:
                private_conv = nn.Sequential(private_layer3, private_layer4)
            else:
                private_conv = nn.Sequential(private_layer3)
        else:
            pass
        # reset private convolution layers
        self.private_conv = private_conv
        '''
        if self.parts_num > 0:
            if backbone=='resnet50':
                self.private_conv[1][0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.private_conv[1][0].conv1.stride = (1, 1)
            self.private_conv[1][0].downsample[0].stride = (1, 1)
        '''
        if self.parts_num > 2:
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
            self.adaptive_pool = nn.AdaptiveMaxPool2d((self.parts_num, 1))
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
                part = l_pool[:,:,i].unsqueeze(2)
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



class MGSEResNet50V3b(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGSEResNet50V3b, self).__init__()
        self.loss = loss

        seres50 = SEResNet50()
        self.common = nn.Sequential(seres50.base[0], seres50.base[1], seres50.base[2])
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            if i==0:
                setattr(self, self.branch_name[i], MGBranchV3b(num_classes, backbone='seresnet50', parts_num=num))
            else:
                setattr(self, self.branch_name[i], MGBranchV3b(num_classes, backbone='seresnet50', parts_num=num, pool_channels=1024))

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



################################################ MGNetV3c ##################################################
# for resnet50 pool_channels = 2048  
class MGBranchV3c(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, pool_channels=2048, embedding_dim=256):
        super(MGBranchV3c, self).__init__()

        self.parts_num = parts_num
        if self.parts_num <= 1:
            self.parts_num = 0
        # private convolution layers
        if backbone=='resnet50':
            res50 = torchvision.models.resnet50(pretrained=True)
            private_conv = res50.layer3
            branch_unit = res50.layer4
        elif backbone=='seresnet50':
            seres50 = SEResNet50()
            private_conv = seres50.base[3]
            branch_unit = seres50.base[4]
        else:
            pass
        # reset private convolution layers
        self.private_conv = private_conv
        if self.parts_num > 0:
            if backbone=='resnet50':
                branch_unit[0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                branch_unit[0].conv1.stride = (1, 1)
            branch_unit[0].downsample[0].stride = (1, 1)
        if self.parts_num > 2:
            if backbone=='resnet50':
                self.private_conv[0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.private_conv[0].conv1.stride = (1, 1)
            self.private_conv[0].downsample[0].stride = (1, 1)
        # global embedding
        self.embedding = nn.Sequential(
            branch_unit,
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        # local embedding
        if self.parts_num > 0:
            #self.adaptive_pool = nn.AdaptiveMaxPool2d((self.parts_num, 1))
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.parts_num):
                local_emb = nn.Sequential(
                    copy.deepcopy(branch_unit),
                    nn.AdaptiveMaxPool2d((1, 1)),
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
            #l_pool = self.adaptive_pool(x) 
            hStrip = x.size(2) // self.parts_num
            for i in range(self.parts_num):
                #part = l_pool[:,:,i].unsqueeze(2)
                part = x[:,:,i*hStrip:(i+1)*hStrip]
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



class MGSEResNet50V3c(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGSEResNet50V3c, self).__init__()
        self.loss = loss

        seres50 = SEResNet50()
        self.common = nn.Sequential(seres50.base[0], seres50.base[1], seres50.base[2])
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            setattr(self, self.branch_name[i], MGBranchV3c(num_classes, backbone='seresnet50', parts_num=num))

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



################################################ MGNetV3d ##################################################
# for resnet50 pool_channels = 2048  
class MGBranchV3d(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', parts_num=0, pool_channels=2048, embedding_dim=256):
        super(MGBranchV3d, self).__init__()

        self.parts_num = parts_num
        if self.parts_num <= 1:
            self.parts_num = 0
        if self.parts_num > 1:
            assert pool_channels % parts_num==0, 'pool_channels(%d), parts_num(%d)'%(pool_channels, parts_num)
            self.slice_len = int(pool_channels // parts_num)
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
            if backbone=='resnet50':
                self.private_conv[0][0].conv2.stride = (1, 1)
            elif backbone=='seresnet50':
                self.private_conv[0][0].conv1.stride = (1, 1)
            self.private_conv[0][0].downsample[0].stride = (1, 1)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
        # global embedding
        self.embedding = nn.Sequential(
            nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        # local embedding
        if self.parts_num > 0:
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.parts_num):
                local_emb = nn.Sequential(
                    nn.Conv2d(self.slice_len, embedding_dim, kernel_size=1, bias=False),
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
        pool = self.adaptive_pool(x) 
        global_feat = self.embedding(pool)
        global_feat = global_feat.view(global_feat.size(0), -1)
        if self.training:
            global_score = self.classifier(global_feat)

        local_feat_group = []
        local_score_group = []
        if self.parts_num > 0:
            for i in range(self.parts_num):
                slic = pool[:,i*self.slice_len:(i+1)*self.slice_len]
                extr = getattr(self, self.local_emb_name[i])                                               
                feat = extr(slic)
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

        
class MGResNet50V3d(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGResNet50V3d, self).__init__()
        self.loss = loss

        res50 = torchvision.models.resnet50(pretrained=True)
        self.common = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2
        )
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            #setattr(self, self.branch_name[i], MGBranchV3d(num_classes, parts_num=num))
            if num > 2:
                setattr(self, self.branch_name[i], MGBranchV3a(num_classes, parts_num=num))
            else:
                setattr(self, self.branch_name[i], MGBranchV3d(num_classes, parts_num=num))

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


class MGSEResNet50V3d(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, branch_stripes=[0,2,3], **kwargs):
        super(MGSEResNet50V3d, self).__init__()
        self.loss = loss

        seres50 = SEResNet50()
        self.common = nn.Sequential(seres50.base[0], seres50.base[1], seres50.base[2])
        self.branch_stripes = branch_stripes
        self.branch_name = {}
        for i, num in enumerate(branch_stripes):
            self.branch_name[i] = 'branch%d'%(i+1)
            #setattr(self, self.branch_name[i], MGBranchV3d(num_classes, backbone='seresnet50', parts_num=num))
            if num > 2:
                setattr(self, self.branch_name[i], MGBranchV3a(num_classes, backbone='seresnet50', parts_num=num))
            else:
                setattr(self, self.branch_name[i], MGBranchV3d(num_classes, backbone='seresnet50', parts_num=num))

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
    net = MGSEResNet50V3d(751, loss={'xent','htri'}, branch_stripes=[0,2,4])
    net.train(False)
    base = net.common
    print(count_num_param(net))
    print(count_num_param(base))
    net.train(True)
    x = torch.Tensor(2,3,384,128)
    y = net(x)
