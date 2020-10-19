# NAIC2020

方案说明：

训练：
sh train.sh
评估：
sh eval.sh
得到最后结果
python ensemble.py 


B榜最终结果用了以下6个模型进行ensemble
model_results = [
    'result_for_ensemble_B/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth.pth',
    'result_for_ensemble_B/resnet101_ibn_b_32x8_90_s1_576x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth.pth',
    'result_for_ensemble_B/resnet101_ibn_b_64x6_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug.pth',
    'result_for_ensemble_B/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug.pth',
    'result_for_ensemble_B/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_std.pth',
    'result_for_ensemble_B/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_allData.pth',
]
模型文件暂未保存。



********************************************************************************************************************************************
思路
## TODO

* Get baseline result(resnet50_bot; mgresnet50v3a)    mgresnet50v3a没收敛
* Delete single pid image            bingo!
* Use all train image                bingo!          线下这样搞 没有验证集 
* DataAugmentation, expecially ColorJittering          RandomPatch？？  有问题
* Align the input image
* Adjust the size of input image   384x128 bingo!  可能可以调？
* Adjust the backbone of model       resnet101  有问题
* triplet loss  arcface    似乎很有作用。
* Get pseudo label of extra data 
* Use AQE and ReRank to postprocess  rerank有作用 AQE还没用。
* Model ensemble
* Apex fp16 bingo!    有加速！！！
* Ranger?
* ibn nolocal gem 是否有用？
* single-gpu vs multi-gpu

next:
1. split the train data and val data
2. all train experiments to single gpu(so one day I can run 8 experiments)
    max_epoch = 90  step = [40, 70]  test_step = 10
   if arcface is useful
3. figure out how to use ibn nolocal gem
4. resnet101
5. dataAug
6. sampler and batchsize
7. align
8. mgn
9. pseudo label, this trick is the last 


## Thought

1. maybe wash data should be the most important things 
2. model :  Efficient-net ibn-b 