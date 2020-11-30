#!/usr/bin/env bash

GPUID=0,1,2,3
#DATASET_DIR=/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/test/0
DATASET_DIR=/data/xuzihao/NAIC/ReID/data/NAIC_2020/extraTrain/REID2019_fusai/fusai_2019_1

CUDA_VISIBLE_DEVICES=$GPUID python getBodyKeypoints.py --cfg 384x288_d256x3_adam_lr1e-3.yaml --dataset $DATASET_DIR
mv ./output/mydataset_read_dir/pose_resnet_152/384x288_d256x3_adam_lr1e-3/_json $DATASET_DIR