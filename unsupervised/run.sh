#!/usr/bin/env bash 


MODEL_FILE=../log/20200914/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth/best_model.pth.tar

#python dealNoLabelData.py --config_type resnet50_bagOfTricks  --dataset ../../data/NAIC_2020/train/noLabelImg --load-weights $MODEL_FILE  --gpu-devices 0,1,2,3

ROOT='../../data/NAIC_2020/train'
SUBSET='noLabelImg'
THRES=0.75

python cluster.py --root $ROOT --subset $SUBSET --thres $THRES