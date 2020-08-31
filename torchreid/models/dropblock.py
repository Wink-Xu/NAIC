import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
import torchvision
from .resnet import weights_init_kaiming, weights_init_classifier
from .drop import DropBlock2D, BatchDrop
import copy
#import ipdb


__all__ = ['DBResNet50', 'DBSEResNet50', 'BFEResNet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'se_resnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, disable_drop=True):
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
        self.dropblock = None
        if not disable_drop:
            self.dropblock = DropBlock2D(keep_prob=0.9, block_size=5)

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

        if self.dropblock is not None:
            out = self.dropblock(out)

        out += residual
        out = self.relu(out)

        return out


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, reduction, stride=1, downsample=None, se_disable=False, disable_drop=True):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        if se_disable==True:
            self.se_module = None
        self.dropblock = None
        if not disable_drop:
            self.dropblock = DropBlock2D(keep_prob=0.9, block_size=5)

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

        if self.se_module is not None:
            out = self.se_module(out)

        if self.dropblock is not None:
            out = self.dropblock(out)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, disable_drop=False) #TRY
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, disable_drop=False) #TRY
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, disable_drop=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, disable_drop=disable_drop))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, disable_drop=disable_drop))

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class SEResNet(nn.Module):

    def __init__(self, block, layers, reduction, dropout_p=0.5, num_classes=1000,
                 downsample_kernel_size=3, downsample_padding=1):
        super(SEResNet, self).__init__()
        self.inplanes = 64

        layer0_modules = [
            ('conv1', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(self.inplanes)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)),
        ]
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(block, 64, layers[0], reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], reduction, stride=2,
            downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding)
        self.layer3 = self._make_layer(block, 256, layers[2], reduction, stride=2,
            downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding, disable_drop=False) #TRY
        self.layer4 = self._make_layer(block, 512, layers[3], reduction, stride=2,
            downsample_kernel_size=downsample_kernel_size, downsample_padding=downsample_padding, disable_drop=False) #TRY

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0, disable_drop=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, reduction, stride, downsample, disable_drop=disable_drop))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction, disable_drop=disable_drop))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def se_resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SEResNet(SEResNetBottleneck, [3, 4, 6, 3], 16, dropout_p=None,
                  downsample_kernel_size=1, downsample_padding=0)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['se_resnet50'])
        model_dict = model.state_dict()
        update_dict = dict()
        for k, v in pretrain_dict.items():
            if k in model_dict and model_dict[k].size() == v.size():
                update_dict[k] = v
            else:
                print(k)
        model_dict.update(update_dict)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


class DBResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(DBResNet50, self).__init__()
        self.loss = loss
        base = resnet50(pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-2])
        self.base[-1][0].conv2.stride = (1, 1)
        self.base[-1][0].downsample[0].stride = (1, 1)
        self.gavg_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
        )
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
        #x = F.avg_pool2d(x, x.size()[2:])
        x = self.gavg_pool(x)
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


class BFEResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, width_ratio=1.0, height_ratio=0.5, **kwargs):
        super(BFEResNet50, self).__init__()
        self.loss = loss
        base = torchvision.models.resnet50(pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-2])
        self.base[-1][0].conv2.stride = (1, 1)
        self.base[-1][0].downsample[0].stride = (1, 1)
        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
        )
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.apply(weights_init_classifier)

        self.b1 = Bottleneck(2048, 512)
        self.drop1 = BatchDrop(width_ratio, height_ratio)
        self.bf1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
        )
        self.bf1.apply(weights_init_kaiming)
        self.bclf1 = nn.Linear(512, num_classes)
        self.bclf1.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
        g_f = F.avg_pool2d(x, x.size()[2:])
        g_f = g_f.view(g_f.size(0), -1)
        g_f = self.bottleneck(g_f)

        b_f1 = self.b1(x)
        b_f1 = self.drop1(b_f1)
        b_f1 = F.max_pool2d(b_f1, b_f1.size()[2:])
        b_f1 = b_f1.view(b_f1.size(0), -1)
        b_f1 = self.bf1(b_f1)

        if not self.training:
            return torch.cat((g_f, b_f1), 1)
        y = self.classifier(g_f)
        y1 = self.bclf1(b_f1)

        if self.loss == {'xent'}:
            return (y, y1)
        elif self.loss == {'xent', 'htri'}:
            return (y, y1), (g_f, b_f1)
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class DBSEResNet50(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(DBSEResNet50, self).__init__()
        self.loss = loss
        base = se_resnet50(pretrained=True)
        self.base = nn.Sequential(*list(base.children())[:-2])
        self.base[-1][0].conv2.stride = (1, 1)
        self.base[-1][0].downsample[0].stride = (1, 1)
        self.gavg_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.bottleneck = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
        )
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.base(x)
        #x = F.avg_pool2d(x, x.size()[2:])
        x = self.gavg_pool(x)
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

    net = se_resnet50(pretrained=True)
    print(count_num_param(net))
    #print(net)
    net.train(True)
    x = torch.Tensor(2,3,224,224)
    y = net(x)
    from IPython import embed
    embed()
