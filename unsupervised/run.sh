#!/usr/bin/env bash 


MODEL_FILE=/data/xuzihao/NAIC/ReID/code/log/20201026/resnet101_ibn_b_32x8_90_s1_384x192_rematch_base/best_model.pth.tar

python dealNoLabelData.py --config_type resnet50_bagOfTricks  --dataset ../../data/NAIC_2020/rematch/unlabel --load-weights $MODEL_FILE  --gpu-devices 6,7

# ROOT='../../data/NAIC_2020/train'
# SUBSET='noLabelImg'
# THRES=0.75

# python cluster.py --root $ROOT --subset $SUBSET --thres $THRES