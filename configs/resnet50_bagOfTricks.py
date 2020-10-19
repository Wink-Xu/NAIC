from config import config


config.config_type = 'resnet50_bagOfTricks'
config.use_cpu = False
config.gpu_devices = 0

## datasets 

config.dataset = 'NAIC_2020' 
config.height = 384
config.width = 192

# Optimization options

# config.optim = 'Ranger'
# config.lr = 0.006               暂时没用 估计是学习率设置问题？

config.optim = 'adam'
config.lr = 0.00035


# 
config.max_epoch = 90
config.train_batch = 32
config.stepsize = [40, 70]

config.save_dir = 'log'

# tricks

config.arch = 'resnet50_bot'
config.resnet101_a = 0
config.resnet101_b = 1
config.efficientnet = 0
config.se_resnet101_ibn_a = 0


config.label_smooth = False
config.class_balance = True

config.num_instance = 8

config.triplet = 1
config.weight_triplet = 0
config.TRI_MARGIN = 0.3
config.CE_LOSS_WEIGHT = 0.33
config.TRI_LOSS_WEIGHT = 1

config.oim = 0

config.center = 0
config.CENTER_LR = 0.5
config.CENTER_LOSS_WEIGHT = 0.0003


config.nl = 0
config.ibn = 1
config.gem = 1 
config.loss_type = 'arcface'
config.COSINE_SCALE = 30.0
config.COSINE_MARGIN = 0.3

config.load_weights = ''
