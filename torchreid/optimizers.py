from __future__ import absolute_import

import torch
from .ranger import Ranger

def init_optim(optim, params, lr, weight_decay):

    if optim == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'amsgrad':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    elif optim == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optimizer: {}".format(optim))


def make_optimizer(config, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = config.lr
        weight_decay = config.weight_decay
        if "bias" in key:
            lr = config.lr 
            weight_decay = config.weight_decay
        if "gap" in key:
            lr = config.lr * 10
            weight_decay = 0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if config.optim == 'SGD':
        optimizer = getattr(torch.optim, config.optim)(params)
    elif config.optim == 'adam':
        optimizer = getattr(torch.optim, "Adam")(params)
    elif config.optim == 'Ranger':
        optimizer = Ranger(params)
    else:
        optimizer = getattr(torch.optim, config.optim)(params)

    return optimizer