import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from .cbam_official import *
from .resnet import weights_init_kaiming, weights_init_classifier
#import ipdb as pdb
from IPython import embed


__all__ = ['CBAResNet50V3OF', 'CBAResNet50V3AOF']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM( planes, 16 )
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM(planes * 4, no_spatial=False)
        else:
            self.cbam = None

    def forward(self, input):
        istuple = isinstance(input, tuple)
        if istuple:
            x, key_pts = input
        else:
            x, key_pts = input, None
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

        if self.cbam is not None:
            out = self.cbam(out, key_pts=key_pts)

        out += residual
        out = self.relu(out)

        if istuple:
            out = (out, key_pts)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes, att_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR 
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        if att_type=='BAM':
            self.bam1 = BAM(64*block.expansion)
            self.bam2 = BAM(128*block.expansion)
            self.bam3 = BAM(256*block.expansion)
        else:
            self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, 64,  layers[0], att_type=att_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type=att_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type=att_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type=att_type)

        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    if self.state_dict()[key].dim() < 2:
                        self.state_dict()[key][...] = 0
                    else:
                        init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key or "SpatialAttn" in key or 'SpatialMultiAttn' in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1, att_type=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=att_type=='CBAM'))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=att_type=='CBAM'))

        return nn.Sequential(*layers)

    def forward(self, input):
        istuple = isinstance(input, tuple)
        if istuple:
            x, key_pts = input
        else:
            x = input

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        if istuple:
            x = (x, key_pts)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        if istuple:
            x = x[0]
        return x

def ResidualNet(network_type, depth, num_classes, att_type):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes, att_type)

    elif depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes, att_type)

    elif depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes, att_type)

    return model


def cba_resnet50(pretrained=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3],  'ImageNet', 1000, 'CBAM')
    if pretrained:
        model_dict = model.state_dict()
        checkpoint = torch.load('/data/xuzihao/ReID/code/RESNET50_CBAM_new_name_wrap.pth')
        pretrain_dict = checkpoint['state_dict']
        update_dict = dict()
        for k, v in pretrain_dict.items():
            k = k.replace('module.', '')
            if k in model_dict and model_dict[k].size() == v.size():
                update_dict[k] = v
        #print(update_dict.keys())
        #print(len(update_dict.keys()))
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_kaiming)
    return model


class CBAResNet50V3OF(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(CBAResNet50V3OF, self).__init__()
        self.loss = loss
        base = cba_resnet50(pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-1])

        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
        )
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, key_pts=None):
        x = self.base(x, key_pts=key_pts)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        f = self.bottleneck(f)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class CBAResNet50V3AOF(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(CBAResNet50V3AOF, self).__init__()
        self.loss = loss
        self.base = cba_resnet50(pretrained=True)
        self.base.layer4[0].conv2.stride = (1, 1)
        self.base.layer4[0].downsample[0].stride = (1, 1)

        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
        )
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.apply(weights_init_classifier)

    def forward(self, input):
        x = self.base(input)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        f = self.bottleneck(f)
        if not self.training:
            return f
        y = self.classifier(f)

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

    net = CBAResNet50V3AOF(1000)
    base = net.base
    print(count_num_param(net), count_num_param(base))
    net.train(True)
    print(net)
    x = torch.Tensor(2,3,224,224)
    key_pts = torch.Tensor([[0.3,0.3],[0.7,0.7]])
    y = net((x, key_pts))
    embed()

    '''
    model = ResNet(Bottleneck, [3, 4, 6, 3],  'ImageNet', 1000, 'CBAM')
    model_dict = model.state_dict()
    print(model_dict.keys())
    print(len(model_dict.keys()))

    checkpoint = torch.load('/home1/chaizh/.torch/models/RESNET50_CBAM_new_name_wrap.pth')
    pretrain_dict = checkpoint['state_dict']
    print(pretrain_dict.keys())
    print(len(pretrain_dict.keys()))

    #pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    update_dict = dict()
    for k, v in pretrain_dict.items():
        k = k.replace('module.', '')
        if k in model_dict and model_dict[k].size() == v.size():
            update_dict[k] = v
    print(update_dict.keys())
    print(len(update_dict.keys()))
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    '''
