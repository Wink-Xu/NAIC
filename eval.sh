#!/usr/bin/env bash 

## get mAP and cmc
# python train_imgreid_xent.py --config_type resnet50_bagOfTricks --dataset NAIC_2020  --load-weight /data/xuzihao/NAIC/ReID/code/log/2020/NAIC/test/checkpoint_ep98.pth.tar --evaluate \
#                               --vis-ranked-res --save-dir log/2020/NAIC/show_Result/res1 --gpu-devices 0 

## get json result   注意修改模型
python eval.py --config_type resnet50_bagOfTricks --dataset NAIC_2020_test  --load-weight /data/xuzihao/NAIC/ReID/code/log/2020/NAIC/resnet50_bot_16x4_120_s2_384x128_ad2/best_model.pth.tar --evaluate \
                              --vis-ranked-res --save-dir log/2020/NAIC/show_Result/resnet50_bot_16x4_120_s2_384x128_ad2_rerank/ --gpu-devices 0,1,2,3
