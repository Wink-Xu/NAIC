# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

class CrossEntropyLossFastReID(nn.Module):
    """
    A class that stores information and compute losses about outputs of a Baseline head.
    """

    def __init__(self, num_classes):
        super(CrossEntropyLossFastReID, self).__init__()
        self._num_classes = num_classes
        self._eps = 0.1
        self._alpha = 0.3
        self._scale = 1

        self._topk = (1,)

    def forward(self, pred_class_logits, _, gt_classes):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
#        self._log_accuracy(pred_class_logits, gt_classes)
        if self._eps >= 0:
            smooth_param = self._eps
        else:
            # adaptive lsr
            soft_label = F.softmax(pred_class_logits, dim=1)
            smooth_param = self._alpha * soft_label[torch.arange(soft_label.size(0)), gt_classes].unsqueeze(1)

        log_probs = F.log_softmax(pred_class_logits, dim=1)
        with torch.no_grad():
            targets = torch.ones_like(log_probs)
            targets *= smooth_param / (self._num_classes - 1)
            targets.scatter_(1, gt_classes.data.unsqueeze(1), (1 - smooth_param))

        loss = (-targets * log_probs).mean(0).sum()
        #import pdb
        #pdb.set_trace()
        return loss * self._scale,
    
		
