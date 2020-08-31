from config import config


config.config_type = 'resnet101_bagOfTricks'
config.use_cpu = False
config.gpu_devices = 0

## datasets 

config.dataset = 'NAIC_2020' 
config.height = 384
config.width = 128

# Optimization options

config.optim = 'adam'
config.lr = 0.00035

# 
config.max_epoch = 120
config.train_batch = 32
config.stepsize = [30, 70]

config.save_dir = 'log'

# tricks

config.arch = 'resnet101_bot'
config.label_smooth = True
config.class_balance = True
