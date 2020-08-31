from __future__ import absolute_import
from __future__ import division

from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torch.nn import functional as F
import torchvision
from .resnet import weights_init_kaiming, weights_init_classifier


"""
Code imported from https://github.com/Cadene/pretrained-models.pytorch
"""


__all__ = ['CBAResNet50', 'CBAResNet50V2', 'CBAResNet50V3']


pretrained_settings = {
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class ChanAttnModule(nn.Module):
    # pool_type: 0 both, 1 avg pooling, 2 max pooling
    def __init__(self, channels, reduction, pool_type=0):
        super(ChanAttnModule, self).__init__()
        assert channels % reduction == 0
        assert pool_type in (0,1,2), 'Unkown pooling type: {0}.'.format(self.pool_type)
        self.pool_type = pool_type
        if self.pool_type==0 or self.pool_type==1:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if self.pool_type==0 or self.pool_type==2:
            self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        if self.pool_type==0:
            x1 = self.avg_pool(x)
            x2 = self.max_pool(x)
            x1 = self.mlp(x1)
            x2 = self.mlp(x2)
            x = x1 + x2
        elif self.pool_type==1:
            x1 = self.avg_pool(x)
            x = self.mlp(x1)
        else:
            x1 = self.max_pool(x)
            x = self.mlp(x1)
        x = self.sigmoid(x)
        return module_input * x


class SpatAttnModule(nn.Module):
    # pool_type: 0 both, 1 avg pooling, 2 max pooling
    def __init__(self, kernel_size=7, pool_type=0):
        super(SpatAttnModule, self).__init__()
        assert pool_type in (0,1,2), 'Unkown pooling type: {0}.'.format(self.pool_type)
        self.pool_type = pool_type
        padding = int(kernel_size) // 2
        if self.pool_type==0:
            self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        else:
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        p_maps = []
        if self.pool_type==0 or self.pool_type==1:
            x1 = x.mean(1, keepdim=True)
            p_maps.append(x1)
        if self.pool_type==0 or self.pool_type==2:
            x1 = x.max(1, keepdim=True)
            p_maps.append(x1[0])
        x = torch.cat(p_maps, dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        return module_input * x


class CBAModule(nn.Module):
    # attn_type: 0 both, 1 cross channels, 2 spatial
    # pool_type: 0 both, 1 avg pooling, 2 max pooling
    def __init__(self, channels, reduction=16, attn_type=0, pool_type=0):
        super(CBAModule, self).__init__()
        assert pool_type in (0,1,2), 'Unkown pooling type: {0}.'.format(self.pool_type)
        assert attn_type in (0,1,2), 'Unkown attention type: {}'.format(self.attn_type) 
        self.attn_type = attn_type
        self.pool_type = pool_type
        if self.attn_type==0 or self.attn_type==1:
            self.channel_attn = ChanAttnModule(channels, reduction, pool_type=pool_type)
        if self.attn_type==0 or self.attn_type==2:
            self.spatial_attn = SpatAttnModule(pool_type=pool_type)
    def forward(self, x):
        if self.attn_type==0 or self.attn_type==1:
            x = self.channel_attn(x)
        if self.attn_type==0 or self.attn_type==2:
            x = self.spatial_attn(x)
        return x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
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

        out = self.cba_module(out) + residual
        out = self.relu(out)

        return out


class CBAResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(CBAResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.cba_module = CBAModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class CBANet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SEANet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(CBANet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def cba_resnet50(num_classes=1000, pretrained='imagenet'):
    model = CBANet(CBAResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    else:
        model.apply(weights_init_kaiming)
    return model


##################### Model Definition #########################


class CBAResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(CBAResNet50, self).__init__()
        self.loss = loss
        base = cba_resnet50()
        self.base = nn.Sequential(*list(base.children())[:-2])
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f
        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class CBAResNet50V2(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(CBAResNet50V2, self).__init__()
        self.loss = loss
        base = cba_resnet50()
        self.base = nn.Sequential(*list(base.children())[:-2])

        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 512),
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


class CBAResNet50V3(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(CBAResNet50V3, self).__init__()
        self.loss = loss
        base = cba_resnet50(pretrained=None)
        self.base = nn.Sequential(*list(base.children())[:-2])

        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
        )
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
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
        #if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            # we ignore the classifier because it is unused at test time
            #num_param -= sum(p.numel() for p in model.classifier.parameters()) / 1e+06
        return num_param

    net = CBAResNet50V3(1000, loss={'xent','htri'})
    print(count_num_param(net), count_num_param(net.base))
    net.train(True)
    x = torch.Tensor(2,3,384,128)
    y = net(x)
