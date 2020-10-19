from easydict import EasyDict

config = EasyDict()


config.use_cpu = False
config.gpu_devices = 0

## datasets 

config.root = '../data'
config.dataset = 'market1501' 
config.workers = 4
config.height = 256
config.width = 128
config.split_id=0

# Optimization options

config.optim = 'adam'
config.max_epoch = 60
config.start_epoch = 0
config.train_batch = 32
config.test_batch = 512
config.lr = 0.0003
config.stepsize = [20, 40]
config.gamma = 0.1
config.weight_decay = 5e-04
config.fixbase_epoch = 0
config.fixbase_lr = 0.0003
config.freeze_bn = False
config.label_smooth = False

# Architecture

config.arch = 'resnet50'

# Miscs
config.print_freq = 10
config.seed = 1
config.resume = ''
config.onnx_export= ''
config.load_weights = ''
config.evaluate = False
config.eval_step = 10
config.start_eval = 0
config.save_dir = 'log'
config.vis_ranked_res = False

config.class_balance = False
config.debug = False
config.use_kpts = False
config.mask_num = 0

config.warmup = 1
config.num_instance = 4

config.apex = 1
config.syncBN = 1

# postprocess
config.rerank = 1
config.aqe = 1
config.TTA = 0
