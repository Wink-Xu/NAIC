from config import config


config.config_type = 'mgresnet50v3a'
config.use_cpu = False
config.gpu_devices = 0

## datasets 

config.dataset = 'market1501' 
config.height = 384
config.width = 128

# Optimization options

config.optim = 'sgd'
config.lr = 0.1

# 
config.max_epoch = 120
config.train_batch = 32
config.stepsize = [30, 70]

config.save_dir = 'log'

# tricks

config.arch = 'mgresnet50v3a'
config.label_smooth = True
config.class_balance = False

config.warmup = 0

