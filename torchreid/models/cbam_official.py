import torch
import math
import torch.nn as nn
import torch.nn.functional as F
#import ipdb as pdb

class BasicCon(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicCon, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def __init__(self, attn='max_avg'):
        super(ChannelPool, self).__init__()
        self.attn = attn

    def forward(self, x):
        if self.attn=='max_avg':
            return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        elif self.attn=='max':
            return torch.max(x,1)[0].unsqueeze(1)
        elif self.attn=='avg':
            return torch.mean(x,1).unsqueeze(1)
        else:
            raise TypeError

class ChannelPooling(nn.Module):
    def __init__(self, groups=1, attn='max_avg'):
        super(ChannelPooling, self).__init__()
        self.attn = attn
        self.groups = int(groups)
        assert self.groups > 0, 'groups > 0'

    def forward(self, x):
        p_maps = []
        g_chs = x.size(1) // self.groups
        for i in range(self.groups):
            x1 = x[:,i*g_chs:(i+1)*g_chs]
            if 'max' in self.attn:
                mp = torch.max(x1,1)[0].unsqueeze(1)
                p_maps.append(mp)
            if 'avg' in self.attn:
                ap = torch.mean(x1,1).unsqueeze(1)
                p_maps.append(ap)
        return torch.cat(p_maps, dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicCon(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class SpatialAttn(nn.Module):
    def __init__(self, attn='max_avg'):
        super(SpatialAttn, self).__init__()
        inplanes = 2 if attn=='max_avg' else 1
        kernel_size = 3

        self.compress = ChannelPool(attn=attn)
        self.conv1 = BasicCon(inplanes, 1, kernel_size, stride=2, padding=(kernel_size-1)//2, relu=False)
        self.conv2 = BasicCon(1, 1, 1, stride=1, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv1(x_compress)
        x_out = F.upsample(x_out, (x_out.size(2)*2, x_out.size(3)*2), mode='bilinear', align_corners=True)
        x_out = self.conv2(x_out)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class Droppart(nn.Module):
    def __init__(self, groups=1, p=0.0):
        super(Droppart, self).__init__()
        self.groups = groups
        self.p = p

    def forward(self, x, key_pts):
        num_kpts = key_pts.size(1)
        n,c,h,w = x.shape
        chs = c // self.groups
        s = int(0.25 * w)

        bkx = key_pts[:,:,0] * w
        bky = key_pts[:,:,1] * h
        print(bkx, bky)
        for i in range(n):
            roll = torch.Tensor(num_kpts).uniform_(0,1)
            bc = 0
            kx, ky = bkx[i], bky[i]
            for j in range(min(self.groups, num_kpts)):
                if roll[j] < self.p and kx[j] >= 0 and ky[j] >= 0:
                    mask = x.new(1, h, w).zero_()
                    cx, cy = kx[j], ky[j]
                    bx, ex = int(max(cx-s, 0)), int(min(cx+s, w))
                    by, ey = int(max(cy-s, 0)), int(min(cy+s, h))
                    mask[0,by:ey,bx:ex] = 1
                    x[i, bc:bc+chs] *= mask
                bc += chs
        return x 


class SpatialMultiAttn(nn.Module):
    def __init__(self, groups=1, attn='max_avg', dpart=0.0):
        super(SpatialMultiAttn, self).__init__()
        self.groups = int(groups)
        inplanes = self.groups 
        inplanes *= 2 if attn=='max_avg' else inplanes
        kernel_size = 3

        if dpart > 0.0:
            self.droppart = Droppart(self.groups, dpart)

        self.compress = ChannelPooling(groups=self.groups, attn=attn)
        self.conv1 = BasicCon(inplanes, self.groups, kernel_size, stride=2, padding=(kernel_size-1)//2, groups=self.groups, relu=False)
        self.conv2 = BasicCon(self.groups, self.groups, 1, stride=1, groups=self.groups, relu=False)

    def forward(self, x, key_pts=None):
        x_compress = self.compress(x)
        if self.training and key_pts is not None:
            x_compress = self.droppart(x_compress, key_pts)

        x_out = self.conv1(x_compress)
        x_out = F.upsample(x_out, (x_out.size(2)*2, x_out.size(3)*2), mode='bilinear', align_corners=True)
        x_out = self.conv2(x_out)
        scale = F.sigmoid(x_out) # broadcasting

        chs = x.size(1) // self.groups
        for i in range(self.groups):
            x[:,i*chs:(i+1)*chs] *= scale[:,i].unsqueeze(1)
        return x


class ChannelMultiAttn(nn.Module):
    def __init__(self, channels, groups=1, reduction_ratio=8, pool_types=['avg', 'max'], dpart=0.0):
        super(ChannelMultiAttn, self).__init__()
        self.groups = int(groups)
        self.pool_types = pool_types
        self.gate_chs = channels // self.groups

        self.attn_conv = nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False)
        if dpart > 0.0:
            self.droppart = Droppart(self.groups, dpart)

        self.mlp_name = {}
        for i in range(self.groups):
            mlp = nn.Sequential(
                Flatten(),
                nn.Linear(self.gate_chs, self.gate_chs // reduction_ratio),
                nn.ReLU(),
                nn.Linear(self.gate_chs // reduction_ratio, self.gate_chs)
                )
            self.mlp_name[i] = 'mlp%d'%(i+1)
            setattr(self, self.mlp_name[i], mlp)

    def forward(self, x, key_pts=None):
        x_trans = self.attn_conv(x)
        if self.training and key_pts is not None:
            x_trans = self.droppart(x_trans, key_pts)

        scales = []
        for i in range(self.groups):
            s_mlp = getattr(self, self.mlp_name[i])
            s_x = x_trans[:,i*self.gate_chs:(i+1)*self.gate_chs]
            channel_att_sum = None
            for pool_type in self.pool_types:
                if pool_type=='avg':
                    avg_pool = F.avg_pool2d(s_x, s_x.size()[2:])
                    channel_att_raw = s_mlp(avg_pool)
                elif pool_type=='max':
                    max_pool = F.max_pool2d(s_x, s_x.size()[2:])
                    channel_att_raw = s_mlp(max_pool)
                elif pool_type=='lp':
                    lp_pool = F.lp_pool2d(s_x, 2, s_x.size()[2:])
                    channel_att_raw = s_mlp(lp_pool)
                elif pool_type=='lse':
                    # LSE pool only
                    lse_pool = logsumexp_2d(s_x)
                    channel_att_raw = s_mlp(lse_pool)

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw
            scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(s_x)
            scales.append(scale)

        scales = torch.cat(scales, 1)
        return x * scales


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            #self.SpatialGate = SpatialGate()
            #self.SpatialAttn = SpatialAttn()
            #self.SpatialMultiAttn = SpatialMultiAttn(groups=8, dpart=0.5)
            #self.SpatialMultiAttn = ChannelMultiAttn(gate_channels, groups=8, reduction_ratio=8, pool_types=pool_types, dpart=0.5)
            self.SpatialMultiAttn = ChannelMultiAttn(gate_channels, groups=8, reduction_ratio=16, pool_types=pool_types, dpart=0.5)

    def forward(self, x, key_pts=None):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            #x_out = self.SpatialGate(x_out)
            #x_out = self.SpatialAttn(x_out)
            x_out = self.SpatialMultiAttn(x_out, key_pts=key_pts)
        return x_out
