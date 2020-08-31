from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .market1501 import Market1501
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .msmt17 import MSMT17
from .dukemtmcvidreid import DukeMTMCVidReID
from .NAIC_2020 import NAIC_2020
from .NAIC_2020_test import NAIC_2020_test
from .VeRi776 import VeRi776
__imgreid_factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'NAIC_2020':NAIC_2020,
    'NAIC_2020_test':NAIC_2020_test,
    'VeRi776':VeRi776
}


__vidreid_factory = {
    'dukemtmcvidreid': DukeMTMCVidReID
}


def get_names():
    return list(__imgreid_factory.keys()) + list(__vidreid_factory.keys())


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)


def init_vidreid_dataset(name, **kwargs):
    if name not in list(__vidreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__vidreid_factory.keys())))
    return __vidreid_factory[name](**kwargs)
