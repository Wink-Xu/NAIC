import torch
import torch.nn as nn
from torch.nn import functional as F
import math, copy
import torch.utils.model_zoo as model_zoo
from .resnet import weights_init_kaiming, weights_init_classifier
#import ipdb


__all__ = ['MaskBV2ResNet50V3', 'MaskBV3ResNet50V3', 'MaskBV3ResNet50V3a', 'MaskBV4ResNet50V3a', 'MaskBV3ResNet50V3aV2', 'MaskResnet50']


model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


def allot_filters(num_filter, num_stripe, ppa):
    attn_size = (num_filter * ppa + num_stripe - 1) // num_stripe
    glob_size = num_filter - attn_size * num_stripe
    return int(attn_size), int(glob_size)

def attn_mask(inp, num_stripe, ppa=1.0):
    num_stripe = int(num_stripe)
    _,c,h,w = inp.shape
    attn_size, glob_size = allot_filters(c, num_stripe, ppa)

    mask = []
    if glob_size > 0:
        mask_g = torch.ones(1, glob_size, h, w)
        mask.append(mask_g)

    stripe = h // num_stripe 
    for i in range(num_stripe):
        mk = torch.zeros(1, attn_size, h, w)
        mk[:,:,i*stripe:(i+1)*stripe] = 1
        mask.append(mk)

    mask = torch.cat(mask, 1)
    if(inp.device.type=='cuda'):
        mask = mask.cuda(inp.device.index)
    return mask

def drop_part(inp, num_stripe, p=0.5, ppa=1.0):
    num_stripe = int(num_stripe)
    n,c,h,w = inp.shape
    part_chs, glob_chs = allot_filters(c, num_stripe, 1.0)

    stripe = h // num_stripe 
    for i in range(n):
        roll = torch.Tensor(num_stripe).uniform_(0,1)
        bc = glob_chs
        br = 0
        for j in range(num_stripe):
            if roll[j] < p:
                inp[i, bc:bc+part_chs, 0:br] = 0
                inp[i, bc:bc+part_chs, br+stripe:] = 0
            bc += part_chs
            br += stripe

    return inp 

class Droppart(nn.Module):
    def __init__(self, groups=1, p=0.5):
        super(Droppart, self).__init__()
        self.groups = groups
        self.p = p

    def forward(self, x):
        n,c,h,w = x.shape
        chs = c // self.groups

        stripe = h // self.groups
        for i in range(n):
            roll = torch.Tensor(self.groups).uniform_(0,1)
            bc = 0
            br = 0
            for j in range(self.groups):
                if roll[j] < self.p:
                    x[i, bc:bc+chs, 0:br] = 0
                    x[i, bc:bc+chs, br+stripe:] = 0
                bc += chs
                br += stripe
        return x 


def pb_drop_part(inp, key_pts, p=0.5, ppa=1.0):
    num_kpts = len(key_pts)
    n,c,h,w = inp.shape
    part_chs, glob_chs = allot_filters(c, num_kpts, 1.0)
    s = int(0.25 * w)

    for i in range(n):
        roll = torch.Tensor(num_kpts).uniform_(0,1)
        bc = glob_chs
        for j in range(num_kpts):
            if roll[j] < p:
                mask = x.new(1, h, w).zero_()
                cx,cy = key_pts[j]
                bx, ex = max(cx-s, 0), min(cx+s, w)
                by, ey = max(cy-s, 0), min(cy+s, h)
                mask[0,by:ey,bx:ex] = 1
                inp[i, bc:bc+part_chs] *= mask
            bc += part_chs

    return inp 


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
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


class MaskBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MaskBottleneck, self).__init__()
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

        self.mask = torch.zeros(1,1,1,1)       

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

        #print('out shape', out.shape)
        #print('mask shape', self.mask.shape)
        if self.mask.shape != out.shape: 
            b,c,h,w = out.shape
            mask0 = torch.zeros(b, c//4, h//2, w)
            mask1 = torch.ones(b, c//4, h-h//2, w)
            mask2 = torch.ones(b, c-2*(c//4), h, w)
            mask01 = torch.cat((mask0, mask1), 2)
            mask10 = torch.cat((mask1, mask0), 2)
            mask = torch.cat((mask01, mask10, mask2), 1)
            if(out.device.type=='cuda'):
                mask = mask.cuda(out.device.index)
            self.mask = mask
            #print('########', self.mask.shape)
        out = out * self.mask

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MaskBottleneckV2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_stripe=2, ppa=0.5):
        super(MaskBottleneckV2, self).__init__()
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

        self.num_stripe = num_stripe 
        self.ppa = ppa

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        mask = attn_mask(out, self.num_stripe, self.ppa)
        out *= mask
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        mask = attn_mask(out, self.num_stripe, self.ppa)
        out *= mask
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        mask = attn_mask(out, self.num_stripe, self.ppa)
        out *= mask
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            mask = attn_mask(out, self.num_stripe, self.ppa)
            residual *= mask

        out += residual
        out = self.relu(out)

        return out


class MaskBottleneckV3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_stripe=2):
        super(MaskBottleneckV3, self).__init__()
        self.num_stripe = num_stripe 
        self.ppa = 1.0
        self.dp = 1.0
        groups = 1

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        #self.drop_part = Droppart(groups=self.num_stripe)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        training = False
        #training = self.training

        out = self.conv1(x)
        out = self.bn1(out)
        if training:
            #out = drop_part(out, self.num_stripe, p=self.dp, ppa=self.ppa)
            out = self.drop_part(out) # TRY
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if training:
            #out = drop_part(out, self.num_stripe, p=self.dp, ppa=self.ppa)
            out = self.drop_part(out) # TRY
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if training:
            #out = drop_part(out, self.num_stripe, p=self.dp, ppa=self.ppa)
            out = self.drop_part(out) # TRY

        if self.downsample is not None:
            residual = self.downsample(x)
            if training:
                #residual = drop_part(residual, self.num_stripe, p=self.dp, ppa=self.ppa)
                residual = self.drop_part(residual) # TRY

        out += residual
        out = self.relu(out)

        return out


# ppa, the proportion of prior attention filters
class PriAttnBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, strips=3, ppa=0.6):
        super(PriAttnBottleneck, self).__init__()
        self.strips = strips
        self.ppa = ppa

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


class ResNet(nn.Module):

    def __init__(self, block, layers, enlarge=1, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if type(block) == tuple:
            assert len(block) > 1, 'len(block) > 1'
            block0 = block[0]
            block = block[1]
        else:
            block0 = block

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block0, 64, layers[0])
        self.layer2 = self._make_layer(block0, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * enlarge, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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


def Resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def MaskResnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(MaskBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def MaskResnet50V2(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet((Bottleneck, MaskBottleneckV2), [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def MaskResnet50V3(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(MaskBottleneckV3, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    else:
        model.apply(weights_init_kaiming)
    return model


def PriAttnResnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


class MaskBV2ResNet50V3(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(MaskBV2ResNet50V3, self).__init__()
        self.loss = loss
        #resnet50 = MaskResnet50(pretrained=True)
        resnet50 = MaskResnet50V2(pretrained=True)
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

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


class MaskBV3ResNet50V3(nn.Module):
    def __init__(self, num_classes, num_stripe=2, pool_channels=2048, embedding_dim=256, loss={'xent'}, **kwargs):
        super(MaskBV3ResNet50V3, self).__init__()

        self.num_stripe = num_stripe
        self.loss = loss
        res50 = Resnet50(pretrained=True)
        masknetv3 = MaskResnet50V3(pretrained=False)

        self.common = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2, res50.layer3
        )
        self.glob = res50.layer4
        self.local = masknetv3.layer4

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
        if self.num_stripe > 0:
            self.adaptive_pool = nn.AdaptiveMaxPool2d((self.num_stripe, 1))
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.num_stripe):
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


    def forward(self, x):
        x = self.common(x)
        x1 = self.glob(x)
        x2 = self.local(x)

        global_feat = self.embedding(x1)
        global_feat = global_feat.view(global_feat.size(0), -1)
        if self.training:
            global_score = self.classifier(global_feat)

        local_feat_group = []
        local_score_group = []
        if self.num_stripe > 0:
            l_pool = self.adaptive_pool(x2) 
            for i in range(self.num_stripe):
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
            scores = tuple([global_score] + local_score_group)
            feats = tuple([global_feat])
            if self.loss == {'xent'}:
                return scores
            elif self.loss == {'xent', 'htri'}:
                return scores, feats 
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            feats = tuple([global_feat] + local_feat_group)
            feature = torch.cat(feats, 1)
            return feature



class MaskBV3ResNet50V3a(nn.Module):
    def __init__(self, num_classes, num_stripe=2, pool_channels=2048, embedding_dim=256, loss={'xent'}, **kwargs):
        super(MaskBV3ResNet50V3a, self).__init__()

        self.num_stripe = num_stripe
        self.loss = loss
        res50 = Resnet50(pretrained=True)
        masknetv3 = MaskResnet50V3(pretrained=False)

        self.common = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2, res50.layer3
        )
        self.glob = res50.layer4
        self.local = masknetv3.layer4

        self.glob[0].conv2.stride = (1, 1)
        self.glob[0].downsample[0].stride = (1, 1)
        self.local[0].conv2.stride = (1, 1)
        self.local[0].downsample[0].stride = (1, 1)

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
        if self.num_stripe > 0:
            self.adaptive_pool = nn.AdaptiveMaxPool2d((self.num_stripe, 1))
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.num_stripe):
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


    def forward(self, x):
        x = self.common(x)
        x1 = self.glob(x)
        x2 = self.local(x)

        global_feat = self.embedding(x1)
        global_feat = global_feat.view(global_feat.size(0), -1)
        if self.training:
            global_score = self.classifier(global_feat)

        local_feat_group = []
        local_score_group = []
        if self.num_stripe > 0:
            l_pool = self.adaptive_pool(x2) 
            for i in range(self.num_stripe):
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
            scores = tuple([global_score] + local_score_group)
            feats = tuple([global_feat])
            if self.loss == {'xent'}:
                return scores
            elif self.loss == {'xent', 'htri'}:
                return scores, feats 
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            feats = tuple([global_feat] + local_feat_group)
            feature = torch.cat(feats, 1)
            return feature


class MaskBV3ResNet50V3aV2(nn.Module):
    def __init__(self, num_classes, num_stripe=2, pool_channels=2048, embedding_dim=256, loss={'xent'}, **kwargs):
        super(MaskBV3ResNet50V3aV2, self).__init__()
        assert pool_channels % num_stripe==0, 'pool_channels(%d) num_stripe(%d)'%(pool_channels, num_stripe)
        self.num_stripe = num_stripe
        self.loss = loss
        self.slice_len = int(pool_channels // num_stripe)
        #self.slice_len = int(pool_channels // num_stripe) * 2  #TRY
        res50 = Resnet50(pretrained=True)
        masknetv3 = MaskResnet50V3(pretrained=False)
        #masknetv3 = MaskResnet50V3(pretrained=False, enlarge=2)  #TRY 

        self.common = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2, res50.layer3
        )
        self.glob = res50.layer4
        self.local = masknetv3.layer4

        self.glob[0].conv2.stride = (1, 1)
        self.glob[0].downsample[0].stride = (1, 1)
        self.local[0].conv2.stride = (1, 1)
        self.local[0].downsample[0].stride = (1, 1)

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
        if self.num_stripe > 0:
            self.adaptive_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.local_emb_name = {}
            self.local_cls_name = {}
            for i in range(self.num_stripe):
                local_emb = nn.Sequential(
                    nn.Conv2d(self.slice_len, embedding_dim, kernel_size=1, bias=False),
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
        x = self.common(x)
        x1 = self.glob(x)
        x2 = self.local(x)

        global_feat = self.embedding(x1)
        global_feat = global_feat.view(global_feat.size(0), -1)
        if self.training:
            global_score = self.classifier(global_feat)

        local_feat_group = []
        local_score_group = []
        if self.num_stripe > 0:
            l_pool = self.adaptive_pool(x2) 
            for i in range(self.num_stripe):
                slic = l_pool[:,i*self.slice_len:(i+1)*self.slice_len]
                extr = getattr(self, self.local_emb_name[i])                                               
                feat = extr(slic)
                feat = feat.view(feat.size(0), -1)
                local_feat_group.append(feat)
                if self.training:
                    cls = getattr(self, self.local_cls_name[i])                                               
                    score = cls(feat)
                    local_score_group.append(score)                                                 

        if self.training:
            scores = tuple([global_score] + local_score_group)
            feats = tuple([global_feat])
            if self.loss == {'xent'}:
                return scores
            elif self.loss == {'xent', 'htri'}:
                return scores, feats 
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            feats = tuple([global_feat] + local_feat_group)
            feature = torch.cat(feats, 1)
            return feature



class MaskBV4ResNet50V3a(nn.Module):
    def __init__(self, num_classes, num_stripe=[2,3], pool_channels=2048, embedding_dim=256, loss={'xent'}, **kwargs):
        super(MaskBV4ResNet50V3a, self).__init__()

        self.num_stripe = num_stripe
        self.loss = loss
        res50 = Resnet50(pretrained=True)
        self.common = nn.Sequential(res50.conv1, res50.bn1, res50.relu,
            res50.maxpool, res50.layer1, res50.layer2, res50.layer3
        )
        glob = res50.layer4
        glob[0].conv2.stride = (1, 1)
        glob[0].downsample[0].stride = (1, 1)
        self.glob = nn.Sequential(glob, nn.AdaptiveMaxPool2d((1, 1)))

        self.branch_name = {}
        for num_s in self.num_stripe:
            masknetv3 = MaskResnet50V3(pretrained=False)
            branch_conv = masknetv3.layer4
            branch_conv[0].conv2.stride = (1, 1)
            branch_conv[0].downsample[0].stride = (1, 1)
            local = nn.Sequential(branch_conv, nn.AdaptiveMaxPool2d((num_s, 1)))
            self.branch_name[num_s] = 'branch%d'%(num_s)
            setattr(self, self.branch_name[num_s], local)

        # global embedding
        self.embedding = nn.Sequential(
            nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.embedding.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        # local embedding
        self.local_emb_name = {}
        self.local_cls_name = {}
        for num_s in self.num_stripe:
            for i in range(num_s):
                local_emb = nn.Sequential(
                    nn.Conv2d(pool_channels, embedding_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(embedding_dim),
                )
                classifier = nn.Linear(embedding_dim, num_classes)
                local_emb.apply(weights_init_kaiming)
                classifier.apply(weights_init_classifier)
                self.local_emb_name['%d_%d'%(num_s, i)] = 'local_emb%d_%d'%(num_s, i+1)
                self.local_cls_name['%d_%d'%(num_s, i)] = 'local_cls%d_%d'%(num_s, i+1)
                setattr(self, self.local_emb_name['%d_%d'%(num_s, i)], local_emb)
                setattr(self, self.local_cls_name['%d_%d'%(num_s, i)], classifier)


    def forward(self, x):
        x = self.common(x)
        x1 = self.glob(x)
        global_feat = self.embedding(x1)
        global_feat = global_feat.view(global_feat.size(0), -1)
        if self.training:
            global_score = self.classifier(global_feat)

        local_feat_group = []
        local_score_group = []
        for num_s in self.num_stripe:
            xb = getattr(self, self.branch_name[num_s])(x)
            for i in range(num_s):
                part = xb[:,:,i].unsqueeze(2)
                extr = getattr(self, self.local_emb_name['%d_%d'%(num_s, i)])                                               
                feat = extr(part)
                feat = feat.view(feat.size(0), -1)
                local_feat_group.append(feat)
                if self.training:
                    cls = getattr(self, self.local_cls_name['%d_%d'%(num_s, i)])                                               
                    score = cls(feat)
                    local_score_group.append(score)                                                 

        if self.training:
            scores = tuple([global_score] + local_score_group)
            feats = tuple([global_feat])
            if self.loss == {'xent'}:
                return scores
            elif self.loss == {'xent', 'htri'}:
                return scores, feats 
            else:
                raise KeyError("Unsupported loss: {}".format(self.loss))
        else:
            feats = tuple([global_feat] + local_feat_group)
            feature = torch.cat(feats, 1)
            return feature



if __name__=='__main__':
    def count_num_param(model):
        num_param = sum(p.numel() for p in model.parameters()) / 1e+06
        return num_param

    net = MaskBV3ResNet50V3aV2(1000)
    print(count_num_param(net))
    x = torch.randn(2,3,256,128)
    y = net(x)
