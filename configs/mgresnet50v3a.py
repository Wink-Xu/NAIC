from config import config


config.config_type = 'mgresnet50v3a'
config.use_cpu = False
config.gpu_devices = 0

## datasets 

config.dataset = 'market1501' 
config.height = 384
config.width = 128

# Optimization options

config.optim = 'SGD'
config.lr = 0.1

# 
config.max_epoch = 120
config.train_batch = 32
config.stepsize = [30, 70]

config.save_dir = 'log'

# tricks

config.arch = 'mgresnet50v3a'
config.label_smooth = True
config.class_balance = True
config.num_instance = 4


config.warmup = 0

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
config.ibn = 0
config.gem = 0
config.loss_type = 'arcface'
config.COSINE_SCALE = 30.0
config.COSINE_MARGIN = 0.3

config.load_weights = ''