from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import ipdb as pdb


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    
    Args:
    - num_classes (int): number of classes.
    - epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, epsilon_factor=1.0):
        """
        Args:
        - inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
        - targets: ground truth labels with shape (num_classes)
        """
        epsilon = self.epsilon * epsilon_factor
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - epsilon) * targets + epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class CrossEntropyWithMaxEntropy(nn.Module):
    def __init__(self, num_classes, epsilon=0.0, lambd=1.0, use_gpu=True):
        super(CrossEntropyWithMaxEntropy, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.lambd = lambd
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets, decay_factor=1.0):
        epsilon = self.epsilon * decay_factor
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - epsilon) * targets + epsilon / self.num_classes
        cross_entropy = (- targets * log_probs).mean(0).sum()

        lambd = self.lambd * decay_factor
        mask = torch.lt(targets, 1)
        non_inputs = inputs[mask]
        non_inputs = non_inputs.view(inputs.size(0), -1)
        non_probs = self.softmax(non_inputs)
        non_log_probs = self.logsoftmax(non_inputs)
        entropy = (- non_probs * non_log_probs).mean(0).sum()
        loss = cross_entropy - lambd * entropy
        return loss


class CrossEntropyWithMaxEntropyV2(nn.Module):
    def __init__(self, num_classes, epsilon=0.0, lambd=1.0, use_gpu=True):
        super(CrossEntropyWithMaxEntropyV2, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.lambd = lambd
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets, epsilon_factor=1.0):
        epsilon = self.epsilon * epsilon_factor
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - epsilon) * targets + epsilon / self.num_classes
        cross_entropy = (- targets * log_probs).mean(0).sum()

        #pdb.set_trace()
        probs = self.softmax(inputs)
        probs = probs * (1 - targets)
        entropy = (- probs * log_probs).mean(0).sum()
        loss = cross_entropy - entropy
        return loss


class DirectRegression(nn.Module):
    def __init__(self, feat_dim, num_classes, use_gpu=True):
        super(DirectRegression, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.weight = Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        self.mse = torch.nn.MSELoss()  # loss

    def forward(self, input, targets):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight)) 
        labels = torch.zeros(cosine.size())
        labels[:] = -1
        targets = labels.scatter_(1, targets.unsqueeze(1).data.cpu().long(), 1)
        if self.use_gpu: targets = targets.cuda()
        loss = self.mse(cosine, targets)
        return loss


if __name__=='__main__':
    inputs = torch.randn(4, 8).cuda()
    labels = torch.Tensor([1, 4, 6, 9]).long()
    loss = DirectRegression(8, 10)
    loss = loss.cuda()

    ret = loss(inputs, labels)
