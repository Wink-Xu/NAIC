from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from .marginal_linear import *
from .circle_layer import Circle
from torchreid.losses.face_loss import Arcface
from .resnet_ibn_a import resnet101_ibn_a
from .resnet_ibn_b import resnet101_ibn_b
import math
__all__ = ['ResNet50_bot', 'ResNet50_bot_circle', 'ResNet101_bot']


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
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
    #    nn.init.constant_(m.bias, 0.0)

## -----------------------------------------------------------------------------------------------------------------------------------	
class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight_freeze=False, bias_freeze=False, weight_init=1.0,
                 bias_init=0.0):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None: self.weight.data.fill_(weight_init)
        if bias_init is not None: self.bias.data.fill_(bias_init)
        self.weight.requires_grad_(not weight_freeze)
        self.bias.requires_grad_(not bias_freeze)


def get_norm(norm, out_channels, num_splits=1, **kwargs):
    """
    Args:
        norm (str or callable):
    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm(out_channels, **kwargs),
        }[norm]
    return norm		
			
class IBN(nn.Module):
    def __init__(self, planes, bn_norm, num_splits):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = get_norm(bn_norm, half2, num_splits)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out			

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel / reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel / reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Non_local(nn.Module):
    def __init__(self, in_channels, bn_norm, num_splits, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            get_norm(bn_norm, self.in_channels, num_splits),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        '''
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z		

class GeneralizedMeanPooling(nn.Module):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)



		
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = get_norm(bn_norm, planes, num_splits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes, num_splits)
        self.relu = nn.ReLU(inplace=True)
#        if with_se:
#            self.se = SELayer(planes, reduction)
#        else:
#            self.se = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, num_splits, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if with_ibn:
            self.bn1 = IBN(planes, bn_norm, num_splits)
        else:
            self.bn1 = get_norm(bn_norm, planes, num_splits)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = get_norm(bn_norm, planes, num_splits)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = get_norm(bn_norm, planes * 4, num_splits)
        self.relu = nn.ReLU(inplace=True)
#        if with_se:
#            self.se = SELayer(planes * 4, reduction)
#        else:
#            self.se = nn.Identity()
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
#        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_more(nn.Module):
    def __init__(self, last_stride = 1, bn_norm="BN", num_splits = 1, with_ibn =0 , with_se=0, with_nl=0, block= Bottleneck, layers = [3, 4, 6, 3], non_layers = [0, 2, 3, 0]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = get_norm(bn_norm, 64, num_splits)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, num_splits, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, num_splits, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, num_splits, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, num_splits, with_se=with_se)

        self.random_init()

				
        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm, num_splits)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", num_splits=1, with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                get_norm(bn_norm, planes * block.expansion, num_splits),
            )

        layers = []
        if planes == 512:
            with_ibn = False
        layers.append(block(self.inplanes, planes, bn_norm, num_splits, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, num_splits, with_ibn, with_se))

        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm, num_splits):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm, num_splits) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm, num_splits) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm, num_splits) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm, num_splits) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
				

class ResNet50_bot_circle(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50_bot_circle, self).__init__()
        self.loss = loss
		
        self.base = ResNet50_more(pretrained=True)
        self.avgpool = GeneralizedMeanPoolingP()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(2048),
        )
        self.bottleneck[0].bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = Circle(in_feat = 2048, num_classes = num_classes)
        self.classifier.apply(weights_init_kaiming)
    def forward(self, x, targets =None):
        x = self.base(x)
        x = self.avgpool(x)
        f = x.view(x.size(0), -1)
        f_after = self.bottleneck(f)
        if not self.training:
            return f_after
        y = self.classifier(f_after, targets)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f_after
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

				
def ResNet50_more(config, pretrained=True,  **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    ibn = config.ibn
    if ibn:
        pretrain_path = './pretrained/r50_ibn_a.pth'
    else:
        pretrain_path = './pretrained/resnet50-19c8e357.pth'
    model = ResNet_more(last_stride = 1, bn_norm="BN", num_splits = 1, with_ibn = ibn , with_se=0, with_nl=config.nl, block= Bottleneck, layers = [3, 4, 6, 3], non_layers = [0, 2, 3, 0])	
    if pretrained:
        if not ibn:            
            state_dict = torch.load(pretrain_path)
	# Remove module.encoder in name
            new_state_dict = {}
            for k in state_dict:
                new_k = k
                if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                    new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict
            print(f"Loading pretrained model from {pretrain_path}")
        else:
            state_dict = torch.load(pretrain_path)  # ibn-net
            # Remove module in name
            new_state_dict = {}
            for k in state_dict:
                new_k = k
                if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                    new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict
            print(f"Loading pretrained model from {pretrain_path}")
        model.load_state_dict(state_dict, strict= False)
    return model				

def ResNet101_ibn_a(config, pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    pretrain_path = './pretrained/resnet101_ibn_a.pth.tar'
   
    model = resnet101_ibn_a(last_stride = 1)
    if pretrained:
        model.load_param(pretrain_path)
        print('Loading pretrained ImageNet model......from {}'.format(pretrain_path))
    return model	

def ResNet101_ibn_b(config, pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    pretrain_path = './pretrained/resnet101_ibn_b.pth'
   
    model = resnet101_ibn_b(last_stride = 1)
    if pretrained:
        model.load_param(pretrain_path)
        print('Loading pretrained ImageNet model......from {}'.format(pretrain_path))
    return model	


def ResNet101_more(config, pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    ibn = config.ibn
    if ibn:
        pretrain_path = './pretrained/resnet101_ibn_a.pth.tar'
    else:
        pretrain_path = './pretrained/resnet50-19c8e357.pth'
    
   
    model = ResNet_more(last_stride = 1, bn_norm="BN", num_splits = 1, with_ibn = config.ibn , with_se=0, with_nl=0, block= Bottleneck, layers = [3, 4, 23, 3], non_layers = [0, 2, 9, 0])	
    if pretrained:
        state_dict = torch.load(pretrain_path)  # ibn-net
        # Remove module in name
        new_state_dict = {}
        for k in state_dict:
            new_k = k
            if new_k in model.state_dict() and (model.state_dict()[new_k].shape == state_dict[k].shape):
                new_state_dict[new_k] = state_dict[k]
        state_dict = new_state_dict
        print(f"Loading pretrained model from {pretrain_path}")
    model.load_state_dict(state_dict, strict= False)
    return model				


class ResNet50_bot(nn.Module):
    def __init__(self, num_classes, config, loss={'xent'},  **kwargs):
        super(ResNet50_bot, self).__init__()
        self.loss_type = config.loss_type
        self.loss = loss
        if config.resnet101_a:
            self.base = ResNet101_ibn_a(config, pretrained=True)
        elif config.resnet101_b:
            self.base = ResNet101_ibn_b(config, pretrained=True)
        else:
            self.base = ResNet50_more(config, pretrained=True)
        if config.gem:
            self.gap = GeneralizedMeanPoolingP()
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
        #
        self.bottleneck =  nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        if config.loss_type == 'arcface':
            print('using {} with s:{}, m: {}'.format(config.loss_type, config.COSINE_SCALE, config.COSINE_MARGIN))
            self.classifier = Arcface(2048, num_classes,
                                        s=config.COSINE_SCALE, m=config.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(2048, num_classes, bias = False)
            self.classifier.apply(weights_init_classifier)
    def forward(self, x, targets =None):
        x = self.base(x)
        x = self.gap(x)
        f = x.view(x.size(0), -1)
        f_after = self.bottleneck(f)
        if not self.training:
            return f_after
        if self.loss_type == 'arcface':
            y = self.classifier(f_after, targets)
        else:
            y = self.classifier(f_after)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet101_bot(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet101_bot, self).__init__()
        self.loss = loss
        self.base = ResNet101_more(pretrained=True)
        self.avgpool = GeneralizedMeanPoolingP()
        self.bottleneck = nn.Sequential(
            nn.BatchNorm1d(2048),
        )
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(2048, num_classes, bias = False)
        self.classifier.apply(weights_init_classifier)
    def forward(self, x, targets =None):
        x = self.base(x)
        x = self.avgpool(x)
        f = x.view(x.size(0), -1)
        f_after = self.bottleneck(f)
        if not self.training:
            return f_after
        y = self.classifier(f_after)

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

    # resnet = torchvision.models.resnet50(pretrained=True)
    # base = nn.Sequential(*list(resnet.children())[:-2])
    # print(count_num_param(resnet), count_num_param(base))

    # net = ResNet50(1000, loss={'xent','htri'})
    # print(count_num_param(net), count_num_param(net.base))
    # net.train(True)
    # x = torch.Tensor(2,3,384,128)
    # y = net(x)
    # from IPython import embed

    # embed()
    #pdb.set_trace()
