#!/usr/bin/env bash 

## get mAP and cmc
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks --dataset NAIC_2020 --load-weight /data/xuzihao/NAIC/ReID/code/log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_moreData_gemlr10_bias0_wloss/best_model.pth.tar --evaluate \
#                              --vis-ranked-res --save-dir log/2020/NAIC/show_Result/res1 --gpu-devices 0,1,2,3

## get json result   注意修改模型 
#python eval.py --config_type resnet50_bagOfTricks --dataset NAIC_2020_test  --load-weight /data/xuzihao/NAIC/ReID/code/log/20201013/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_allData/best_model.pth.tar --evaluate \
#                              --vis-ranked-res --save-dir log/20201013/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_allData_90 --gpu-devices 4,5,6,7
### 一定要记得改模型的参数！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

###  image_B

#python eval.py --config_type resnet50_bagOfTricks --dataset NAIC_2020_test_B  --load-weight /data/xuzihao/NAIC/ReID/code/log/20201013/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_allData/best_model.pth.tar --evaluate \
#                              --vis-ranked-res --save-dir log/B/20201013/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_allData_90 --gpu-devices 0,1,2,3,4,5,6,7
### 一定要记得改模型的参数！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！

# #1.
# python eval.py --config_type resnet50_bagOfTricks --dataset NAIC_2020_test_B  --load-weight /data/xuzihao/NAIC/ReID/code/log/20200914/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth/best_model.pth.tar --evaluate \
#                               --vis-ranked-res --save-dir log/B/20200914/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth --gpu-devices 0,1,2,3,4,5,6,7
# #2.
# python eval.py --config_type resnet50_bagOfTricks --dataset NAIC_2020_test_B  --load-weight /data/xuzihao/NAIC/ReID/code/log/20200923/resnet101_ibn_b_32x8_90_s1_576x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth/best_model.pth.tar --evaluate \
#                               --vis-ranked-res --save-dir log/B/20200923/resnet101_ibn_b_32x8_90_s1_576x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth --gpu-devices 0,1,2,3,4,5,6,7
# #3.
# python eval.py --config_type resnet50_bagOfTricks --dataset NAIC_2020_test_B  --load-weight /data/xuzihao/NAIC/ReID/code/log/20200914/resnet101_ibn_b_64x6_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug/best_model.pth.tar --evaluate \
#                               --vis-ranked-res --save-dir log/B/20200914/resnet101_ibn_b_64x6_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug --gpu-devices 0,1,2,3,4,5,6,7
# #4.
# python eval.py --config_type resnet50_bagOfTricks --dataset NAIC_2020_test_B  --load-weight /data/xuzihao/NAIC/ReID/code/log/20200914/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug/best_model.pth.tar --evaluate \
#                               --vis-ranked-res --save-dir log/B/20200914/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug --gpu-devices 0,1,2,3,4,5,6,7
#5.
python eval.py --config_type resnet50_bagOfTricks --dataset NAIC_2020_test_B  --load-weight /data/xuzihao/NAIC/ReID/code/log/20200917/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_std/best_model.pth.tar --evaluate \
                              --vis-ranked-res --save-dir log/B/20200917/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_std --gpu-devices 0,1,2,3,4,5,6,7



