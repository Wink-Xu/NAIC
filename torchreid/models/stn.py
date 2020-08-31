from __future__ import absolute_import
from __future__ import division

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import numpy as np
import scipy.linalg
from .resnet import weights_init_kaiming, weights_init_classifier
#import ipdb


__all__ = ['STN', 'ICSTN', 'STNDN']


# build Spatial Transformer Network with densenet as backbone
class STNDN(nn.Module):
    def __init__(self, inplanes, opt):
        super(STNDN, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True)
        self.base = densenet121.features
        self.feat_dim = 1024

        self.fc1 = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(512, opt.warpDim)
        self.fc1.apply(weights_init_kaiming)
        self.fc2.apply(weights_init_classifier)
        self.opt = opt

    def net_forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        p = self.net_forward(x) 
        pMtrx = vec2mtrx(p, self.opt.warpType)
        warpX = transformImage(x, pMtrx, self.opt.refMtrx, self.opt.W, self.opt.H)
        #print(p[0])

        return warpX, p


# build Spatial Transformer Network
class STN(nn.Module):
    def __init__(self, inplanes, opt):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 4, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=7, stride=1, padding=3)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(8*(opt.W//4*opt.H//4), 48)
        self.fc2 = nn.Linear(48, opt.warpDim)
        self.relu = nn.ReLU(inplace=True)
        initialize(self, opt.stdGP, last0=True)
        self.opt = opt

    def net_forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        #ipdb.set_trace()
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        p = self.net_forward(x) 
        pMtrx = vec2mtrx(p, self.opt.warpType)
        warpX = transformImage(x, pMtrx, self.opt.refMtrx, self.opt.W, self.opt.H)
        #print(p[0])

        return warpX, p


# build Inverse Compositional STN
class ICSTN(nn.Module):
    def __init__(self, inplanes, opt):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 4, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=7, stride=1, padding=3)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(8*(opt.W/4*opt.H/4), 48)
        self.fc2 = nn.Linear(48, opt.warpDim)
        self.relu = nn.ReLU(inplace=True)
        initialize(self, opt.stdGP, last0=True)
        self.opt = opt

    def net_forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, x, p):
        #warpAll = []
        for i in range(self.opt.warpN):
            pMtrx = vec2mtrx(p, self.opt.warpType)
            warpX = transformImage(x, pMtrx, self.opt.refMtrx, self.opt.W, self.opt.H)
            #warpAll.append(warpX)
            dp = self.net_forward(warpX) 
            p = compose(p, dp, self.opt.warpType)
        pMtrx = vec2mtrx(p, self.opt.warpType)
        warpX = transformImage(x, pMtrx, self.opt.refMtrx, self.opt.W, self.opt.H)
        #warpAll.append(warpX)

        return warpX, p


# initialize weights/biases
def initialize(model, stddev, last0=False):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, stddev)
            m.bias.data.normal_(0, stddev)
        elif isinstance(m, nn.Linear): 
            if last0 and m is model.fc2:
                m.weight.data.zero_()
                m.bias.data.zero_()
            else:
                m.weight.data.normal_(0, stddev)
                m.bias.data.normal_(0, stddev)


##################################### warp #####################################
# fit (affine) warp between two sets of points 
def fit(Xsrc, Xdst):
    ptsN = len(Xsrc)
    X,Y,U,V,O,I = Xsrc[:,0],Xsrc[:,1],Xdst[:,0],Xdst[:,1],np.zeros([ptsN]),np.ones([ptsN])
    A = np.concatenate((np.stack([X,Y,I,O,O,O],axis=1),
                        np.stack([O,O,O,X,Y,I],axis=1)),axis=0)
    b = np.concatenate((U,V),axis=0)
    p1,p2,p3,p4,p5,p6 = scipy.linalg.lstsq(A,b)[0].squeeze()
    pMtrx = np.array([[p1,p2,p3],[p4,p5,p6],[0,0,1]],dtype=np.float32)
    return pMtrx

# compute composition of warp parameters
def compose(p, dp, warpType):
    pMtrx = vec2mtrx(p, warpType)
    dpMtrx = vec2mtrx(dp, warpType)
    pMtrxNew = dpMtrx.matmul(pMtrx)
    pMtrxNew /= pMtrxNew[:, 2:3, 2:3]
    pNew = mtrx2vec(pMtrxNew, warpType)
    return pNew

# compute inverse of warp parameters
def inverse(p, warpType):
    pMtrx = vec2mtrx(p, warpType)
    pInvMtrx = pMtrx.inverse()
    pInv = mtrx2vec(pInvMtrx, warpType)
    return pInv

# convert warp parameters to matrix
def vec2mtrx(p, warpType):
    O = torch.zeros(p.size(0), dtype=p.dtype, device=p.device) 
    I = torch.ones(p.size(0), dtype=p.dtype, device=p.device) 
    #O = util.toTorch(np.zeros([opt.batchSize],dtype=np.float32))
    #I = util.toTorch(np.ones([opt.batchSize],dtype=np.float32))
    if warpType=="translation":
        tx,ty = torch.unbind(p,dim=1)
        pMtrx = torch.stack([torch.stack([I,O,tx],dim=-1),
                             torch.stack([O,I,ty],dim=-1),
                             torch.stack([O,O,I],dim=-1)],dim=1)
    if warpType=="similarity":
        pc,ps,tx,ty = torch.unbind(p,dim=1)
        pMtrx = torch.stack([torch.stack([I+pc,-ps,tx],dim=-1),
                             torch.stack([ps,I+pc,ty],dim=-1),
                             torch.stack([O,O,I],dim=-1)],dim=1)
    if warpType=="affine":
        p1,p2,p3,p4,p5,p6 = torch.unbind(p,dim=1)
        pMtrx = torch.stack([torch.stack([I+p1,p2,p3],dim=-1),
                             torch.stack([p4,I+p5,p6],dim=-1),
                             torch.stack([O,O,I],dim=-1)],dim=1)
    if warpType=="homography":
        p1,p2,p3,p4,p5,p6,p7,p8 = torch.unbind(p,dim=1)
        pMtrx = torch.stack([torch.stack([I+p1,p2,p3],dim=-1),
                             torch.stack([p4,I+p5,p6],dim=-1),
                             torch.stack([p7,p8,I],dim=-1)],dim=1)
    return pMtrx

# convert warp matrix to parameters
def mtrx2vec(pMtrx, warpType):
    [row0,row1,row2] = torch.unbind(pMtrx,dim=1)
    [e00,e01,e02] = torch.unbind(row0,dim=1)
    [e10,e11,e12] = torch.unbind(row1,dim=1)
    [e20,e21,e22] = torch.unbind(row2,dim=1)
    if warpType=="translation": p = torch.stack([e02,e12],dim=1)
    if warpType=="similarity": p = torch.stack([e00-1,e10,e02,e12],dim=1)
    if warpType=="affine": p = torch.stack([e00-1,e01,e02,e10,e11-1,e12],dim=1)
    if warpType=="homography": p = torch.stack([e00-1,e01,e02,e10,e11-1,e12,e20,e21],dim=1)
    return p


def toTorch(nparray, device):
    tensor = torch.Tensor(nparray, device=device)
    if device.type is not 'cpu':
        tensor = tensor.cuda(device.index)
    return torch.autograd.Variable(tensor, requires_grad=False)

# warp the image
def transformImage(image, pMtrx, refMtrx, W, H):
    batchSize = pMtrx.size(0)
    refMtrx = toTorch(refMtrx, pMtrx.device)
    #refMtrx = refMtrx.repeat(opt.batchSize, 1, 1)
    refMtrx = refMtrx.repeat(batchSize, 1, 1)
    transMtrx = refMtrx.matmul(pMtrx)
    # warp the canonical coordinates
    X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
    X, Y = X.flatten(), Y.flatten()
    XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T
    XYhom = np.tile(XYhom, [batchSize, 1, 1]).astype(np.float32)
    XYhom = toTorch(XYhom, pMtrx.device)
    XYwarpHom = transMtrx.matmul(XYhom)
    XwarpHom, YwarpHom, ZwarpHom = torch.unbind(XYwarpHom, dim=1)
    Xwarp = (XwarpHom/(ZwarpHom+1e-8)).view(batchSize, H, W)
    Ywarp = (YwarpHom/(ZwarpHom+1e-8)).view(batchSize, H, W)
    grid = torch.stack([Xwarp,Ywarp], dim=-1)
    # sampling with bilinear interpolation
    imageWarp = torch.nn.functional.grid_sample(image, grid, mode="bilinear")
    return imageWarp


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.warpDim = 6
    opt.stdGP = 0.1
    opt.warpType = 'affine'
    opt.refMtrx = np.eye(3).astype(np.float32)
    opt.W = 28 
    opt.H = 28

    model = STN(3, opt)
    x = torch.rand(8, 3, 28, 28)
    y = model(x)
