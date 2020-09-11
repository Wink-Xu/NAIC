#!/usr/bin/env bash 

## get mAP and cmc
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks --dataset NAIC_2020 --load-weight /data/xuzihao/NAIC/ReID/code/log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_moreData_gemlr10_bias0_wloss/best_model.pth.tar --evaluate \
#                              --vis-ranked-res --save-dir log/2020/NAIC/show_Result/res1 --gpu-devices 0,1,2,3

## get json result   注意修改模型 
python eval.py --config_type resnet50_bagOfTricks --dataset NAIC_2020_test  --load-weight /data/xuzihao/NAIC/ReID/code/log/20200910/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew/checkpoint_ep80.pth.tar --evaluate \
                              --vis-ranked-res --save-dir log/20200910/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew --gpu-devices 0,1,2,3
### 一定要记得改模型的参数！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！