from __future__ import absolute_import

from .resnet import *
from .resnext import *
from .seresnet import *
from .densenet import *
from .mudeep import *
from .hacnn import *
from .squeeze import *
from .mobilenetv2 import *
from .shufflenet import *
from .xception import *
from .inceptionv4 import *
from .nasnet import *
from .inceptionresnetv2 import *
from .mgnet import *
from .mgnetv2 import *
from .mgnetv3 import *
from .cbam import *
from .cbam_resnet import *
from .prior_attn import *
from .stn import *
from .part_guided import *
from .dropblock import *
from .sampart import *
from .xception import *
from .hump_mgnetv3 import *
from .resnet50_enhanced import *

__model_factory = {
    'stn': STN,
    'stndn': STNDN,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'seresnet50': SEResNet50,
    'seresnet101': SEResNet101,
    'seresnet101v3': SEResNet101V3,
    'seresnext50': SEResNeXt50,
    'seresnext101': SEResNeXt101,
    'resnext101': ResNeXt101_32x4d,
    'resnet50m': ResNet50M,
    'densenet121': DenseNet121,
    'squeezenet': SqueezeNet,
    'mobilenetv2': MobileNetV2,
    'shufflenet': ShuffleNet,
    'xception': Xception,
    'inceptionv4': InceptionV4,
    'nasnsetmobile': NASNetAMobile,
    'inceptionresnetv2': InceptionResNetV2,
    'mudeep': MuDeep,
    'hacnn': HACNN,
    'mgresnet50': MGResNet50,
    'mgresnet50v2': MGResNet50V2,
    'mgresnet50v3': MGResNet50V3,
    'mgresnet50v3a': MGResNet50V3a,
    'mgresnet50v3x':MGResNet50V3x,
    'mgresnet50v3x_1':MGResNet50V3x_1,
    'mgresnet50v3a_ibn_a':MGResNet50V3a_ibn_a,
    'mgresnet50v3a_ibn_b':MGResNet50V3a_ibn_b,
    'cbamgresnet50v3a':CBAMGResNet50V3a,
    'mgseresnet50v3': MGSEResNet50V3,
    'mgseresnet50v3a': MGSEResNet50V3a,
    'mgseresnet50v3b': MGSEResNet50V3b,
    'mgseresnet50v3c': MGSEResNet50V3c,
    'mgseresnet50v3d': MGSEResNet50V3d,
    'humpmgseresnet50v3a': HumpMGSEResNet50V3a,
    'pcresnet50': PCResnet50,
    'resnet50v2': ResNet50V2,
    'resnet50v3': ResNet50V3,
    'resnet50v3ms': ResNet50V3MS,
    'resnet50_bot': ResNet50_bot,
    'resnet50_bot_circle': ResNet50_bot_circle,
    'resnet101_bot': ResNet101_bot,
    'resnet50v3alg': ResNet50V3aLg,
    'resnet101v3a': ResNet101V3a,
    'seresnet50v2': SEResNet50V2,
    'seresnet50v3': SEResNet50V3,
    'seresnet50v3a': SEResNet50V3a,
    'seresnet50watrousv3': SEResNet50WAtrousV3,
    'cbaresnet50': CBAResNet50,
    'cbaresnet50v2': CBAResNet50V2,
    'cbaresnet50v3': CBAResNet50V3,
    'cbaresnet50v3of': CBAResNet50V3OF,
    'cbaresnet50v3aof': CBAResNet50V3AOF,
    'maskbv2resnet50v3': MaskBV2ResNet50V3,
    'maskbv3resnet50v3': MaskBV3ResNet50V3,
    'maskbv3resnet50v3a': MaskBV3ResNet50V3a,
    'maskbv3resnet50v3av2': MaskBV3ResNet50V3aV2,
    'maskbv4resnet50v3a': MaskBV4ResNet50V3a,
    'pgresnet50': PGResNet50,
    'pgseresnet50': PGSEResNet50,
    'dbresnet50': DBResNet50,
    'dbseresnet50': DBSEResNet50,
    'bferesnet50': BFEResNet50,
    'spresnet50': SPResNet50,
    'spresnet50v2': SPResNet50V2,
    'spresnet50v3a': SPResNet50V3a,
    'spseresnet50v3a': SPSEResNet50V3a,
    'xceptionnet': XceptionNet,
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)
