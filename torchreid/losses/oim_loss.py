# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn

import torch.nn.functional as F
from torch import nn, autograd


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class OIM(autograd.Function):
    def __init__(self, lut, queue, index, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.queue = queue
        self.momentum = momentum
        self.index = index

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs_labeled = inputs.mm(self.lut.t())
        outputs_unlabeled = inputs.mm(self.queue.t())
        return  torch.cat((outputs_labeled, outputs_unlabeled), 1)

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        # used=[]
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat((self.lut, self.queue), 0))
        for x, y in zip(inputs, targets):
            if y<0 or y>=self.lut.size(0):
                self.queue[self.index, :] = x.view(1,-1)
                self.index = (self.index+1) % self.queue.size(0)
            else:
                # if y in used:
                #     continue
                # used.append(y)
                self.lut[y] = self.momentum * self.lut[y] + (1. - self.momentum) * x
                self.lut[y] /= self.lut[y].norm()
        return grad_inputs, None


def oim(inputs, targets, lut, queue, index, momentum=0.5):
    return OIM(lut, queue, index, momentum=momentum)(inputs, targets)

class OIMLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, queue_size=2000, scalar=1.0, momentum=0.5,
                 label_smooth=False, epsilon=0.1, weight=None, reduction='mean', loss_weight=1.0):
        super(OIMLoss, self).__init__()
        self.feat_dim = feat_dim
        # num_classes = num_classes-1
        print(num_classes)
        self.num_classes = num_classes
        self.queue_size = queue_size
        self.momentum = momentum
        self.scalar = scalar
        self.index = 0
        self.loss_weight = loss_weight
        # if weight is None:
        #     self.weight = torch.cat([torch.ones(num_classes).cuda(), torch.zeros(queue_size).cuda()])
        # else:
        #     self.weight = weight
        self.reduction = reduction
        self.label_smooth = label_smooth
        if self.label_smooth:
            self.epsilon = epsilon
            self.logsoftmax = nn.LogSoftmax(dim=1)

        self.register_buffer('lut', torch.zeros(num_classes, feat_dim))
        self.register_buffer('queue', torch.zeros(queue_size, feat_dim))
        self.lut = self.lut.cuda()
        self.queue = self.queue.cuda()

    def forward(self, inputs, targets, normalize_feature=True, margin=0.0):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        # import pdb
        # pdb.set_trace()
        inputs = oim(inputs, targets, self.lut, self.queue, self.index, momentum=self.momentum)
        # targets[targets>=self.num_classes] = self.num_classes
        ### add margin (11/1)
        phi = inputs - margin
        one_hot = torch.zeros(inputs.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1,1).long(), 1)
        inputs = (one_hot*phi) + ((1-one_hot)*inputs)

        inputs *= self.scalar
        ### add label smooth (11/4)
        if self.label_smooth:
            log_probs = self.logsoftmax(inputs)
            targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
            targets = targets.cuda()
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
            loss = (- targets * log_probs).mean(0).sum()
        else:
            # import pdb
            # pdb.set_trace()
            # weight = torch.cat([torch.ones(self.num_classes), torch.zeros(self.queue_size)])
            weight = torch.cat([torch.ones(4768), torch.zeros(self.num_classes+self.queue_size-4768)])
            weight = weight.cuda()
            loss = F.cross_entropy(inputs, targets, weight=weight, reduction=self.reduction, ignore_index=-1)
        #self.index = (self.index + torch.nonzero(targets<0) % self.queue_size
        self.index = (self.index + torch.nonzero(targets>=self.num_classes).size(0)) % self.queue_size
        return loss, inputs