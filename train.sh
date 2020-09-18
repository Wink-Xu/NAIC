#!/usr/bin/env bash 
#1.  resnet50_bot_128_250 
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 128 --save-dir log/2020/NAIC/resnet50_bot_128_250 --max-epoch 250 --gpu-devices 0,1,2,3
#2.  mgresnet50v3a_bot_16x4_250 sampler2
#python train_imgreid_xent.py --config_type mgresnet50v3a  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/mgresnet50v3a_bot_32x4_250_s2 --max-epoch 250 --gpu-devices 4,5
#3. resnet50_bot_64_120 sampler2
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_bot_16x4_120_s2 --gpu-devices 0,1
#4. resnet50_bot_64_120_384x128  sampler2
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_bot_16x4_120_s2_384x128 --gpu-devices 2,3
#5.  mgresnet50v3a_bot_16x4_120_384x128 sampler2 lr 0.1 nowarmup
#python train_imgreid_xent.py --config_type mgresnet50v3a  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/mgresnet50v3a_bot_16x4_120_s2_384x128 --gpu-devices 6,7
#6. resnet50_bot_64_120_384x128  sampler2 _ allData delete <=2 pid pic
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_bot_16x4_120_s2_384x128_ad2 --gpu-devices 0,1
#7. mgresnet50v3a_bot_64_120_384x128  sampler2 _ allData delete <=2 pid pic
#python train_imgreid_xent.py --config_type mgresnet50v3a  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/mgresnet50v3a_bot_16x4_120_s2_384x128_ad2 --gpu-devices 2,3
#8. resnet50_bot_64_120_384x128  sampler2 _ allData delete <=2 pid pic + cj0.5
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_bot_16x4_120_s2_384x128_ad2_cj0.5 --gpu-devices 6,7
#python train_imgreid_xent.py --config_type mgresnet50v3a  --dataset NAIC_2020  --train-batch 64 --save-dir log/test --gpu-devices 4,5
#9. mgresnet50v3a_bot_64_120_384x128  sampler3 _ allData delete <=2 pid pic
#python train_imgreid_xent.py --config_type mgresnet50v3a  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/mgresnet50v3a_bot_16x4_120_s3_384x128_ad2_cj0.5 --gpu-devices 4,5
#10. mgresnet50v3a_bot_64_120_384x128  sampler2 _ allData delete <=3 pid pic
#python train_imgreid_xent.py --config_type mgresnet50v3a  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/mgresnet50v3a_bot_16x4_120_s2_384x128_ad3100_5000 --gpu-devices 2,3
#11. resnet50_bot_64_120_384x128  sampler2 _ allData delete <=2 pid pic + cj0.5 + aug
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_bot_16x4_120_s2_384x128_ad2_aug --gpu-devices 6,7
#12. resnet50_bot_64_120_384x128  sampler2 _ allData delete <=2 pid pic + cj0.5 + resnet101
#python train_imgreid_xent.py --config_type resnet101_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet101_bot_16x4_120_s2_384x128_ad2_aug --gpu-devices 4,5
#13. mgresnet50v3a_bot_64_120_384x128  sampler2 _ allData delete <=3 pid pic
#python train_imgreid_xent.py --config_type mgresnet50v3a  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/mgresnet50v3a_bot_16x4_120_noS_384x128_ad2_aug --gpu-devices 0,1
#14. resnet50_ibna_nl_64_120_384x128  sampler2 _ allData delete <=2 pid pic + cj0.5 + aug
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibna_nl_16x4_120_s2_384x128_ad2_aug --gpu-devices 6,7
#15. resnet50_ibna_64_120_384x128  sampler2 _ allData delete <=2 pid pic + cj0.5 + aug
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibna_16x4_120_s2_384x128_ad2_aug --gpu-devices 4,5
#16. resnet50_ibna_64_120_384x128  sampler2 _ allData delete <=2 pid pic + cj0.5 + aug triplet
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibna_16x4_120_s2_384x128_ad2_aug_triplet --gpu-devices 2,3
#17. mgresnet50v3a_bot_64_120_384x128  sampler3 _ allData delete <=3 pid pic  cj0.5 + aug triplet
#python train_imgreid_xent.py --config_type mgresnet50v3a  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/mgresnet50v3a_bot_16x4_120_S3_384x128_ad2_aug_adam --gpu-devices 0,1
#18. resnet50_ibna_nl_64_120_384x128  sampler2 _ allData delete <=2 pid pic
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibna_nl_16x4_120_s2_384x128_ad2 --gpu-devices 6,7
#19. resnet50_ibna_64_120_384x128  sampler2 _ allData delete <=2 pid pic triplet
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibna_16x4_120_s2_384x128_ad2_triplet --gpu-devices 0,1
#20. resnet50_ibna_64_120_384x128  sampler2 _ allData delete <=2 pid pic triplet arcface
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibna_16x4_120_s2_384x128_ad2_triplet_arcface --gpu-devices 0,1
#21. resnet50_ibna_64_120_384x128  sampler2 _ allData delete <=2 pid pic triplet no ibna
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibna_16x4_120_s2_384x128_ad2_triplet_gpu1--gpu-devices 7
#22. resnet50_ibna_64_120_384x128  sampler2 _ allData delete <=2 pid pic triplet with ibna
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibna_16x4_120_s2_384x128_ad2_triplet_gpu1_ibna --gpu-devices 6
#23. resnet50_ibna_64_120_384x128  sampler2 _ allData delete <=2 pid pic triplet with ibna
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibna_16x4_120_s2_384x128_ad2_triplet_gpu1_apex --gpu-devices 5
##  arcface noibn(single gpu) ibn(single gpu)  ibn-apex(single gpu) 
##  
# ---------------------------------------------------------------------- 20200902
## 5个实验 20200902   newData
## 1. single gpu arcface .  resnet50    mAP: 39.5%  Rank-1  : 60.5%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_16x4_120_s2_384x128_ad2_triplet_gpu1_apex_arcface --gpu-devices 4
## 2. single gpu arcface .  resnet50 + ibn  mAP: 46.0% Rank-1  : 65.0%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_120_s2_384x128_ad2_triplet_gpu1_apex_arcface --gpu-devices 3
## 3. single gpu arcface .  resnet50 + ibn + nl   mAP: 45.9% Rank-1  : 66.0%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_nl_16x4_120_s2_384x128_ad2_triplet_gpu1_apex_arcface --gpu-devices 2
## 4. single gpu arcface .  resnet50 + ibn + nl + gem (gem 学习率问题) mAP: 45.1% Rank-1  : 63.4%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_nl_gem_16x4_120_s2_384x128_ad2_triplet_gpu1_apex_arcface --gpu-devices 1
## 5. single gpu arcface .  resnet50 + sampler1 mAP: 39.8% Rank-1  : 60.3%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_16x4_120_s1_384x128_ad2_triplet_gpu1_apex_arcface --gpu-devices 0
## conclusion  
##  ibn_a 很有用， 过拟合严重。
# ---------------------------------------------------------------------- 20200903
## 实验1. single gpu arcface .  resnet50 + ibn  + data  arcface Apex   mAP: 75.9% Rank-1  : 87.9%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_moreData --gpu-devices 0
## 实验2. single gpu arcface .  resnet50 + ibn  + data  arcface noApex mAP: 75.7% Rank-1  : 88.2%
##python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_arcface_moreData --gpu-devices 1
## 实验3. single gpu arcface .  resnet50 + ibn  + data  noApex no arcface mAP: 73.0% Rank-1  : 85.4%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_moreData --gpu-devices 2
## 实验4. single gpu arcface .  resnet50 + ibn  + data  noApex arcface + cj0.5 mAP: 74.6% Rank-1  : 86.4%
##python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_arcface_moreData_cj0.5 --gpu-devices 3
## 实验5. single gpu arcface .  resnet50 + ibn  + data  noApex arcface + gem + cj0.5  mAP: 73.9% Rank-1  : 86.2%
##python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_arcface_moreData_gem --gpu-devices 4
##实验6. single gpu arcface .  resnet50 + ibn  + data  noApex arcface + gem(lr*10) + cj0.5 mAP: 75.2% Rank-1  : 87.3%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_arcface_moreData_gem10 --gpu-devices 5
## 实验7. single gpu arcface .  resnet50 + ibn  + data  arcface noApex + cj0.5 16 * 6mAP: 76.4% Rank-1  : 87.8%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 96 --save-dir log/2020/NAIC/resnet50_ibn_16x6_90_s1_384x128_ad2_triplet_gpu2_apex_arcface_moreData --gpu-devices 6,7
### conclution  apex有用 快了40分钟。好像也没多少啊 - - 一共9小时左右
###             arcface有用 涨了3个点
###             cj0.5真没用 不知道咋用
###             gem单用没用 但是配上学习率*10有用 涨了1个点 所以接下来实验是 nocj
###             16*6有用涨了1.8个点
###      所以 apex arcface gemlr*10 16*6 可以常用 但是16*6需要两个GPU  这个可以后面再用
## 实验1. single gpu arcface .  resnet50 + ibn  + data  arcface Apex   mAP: 75.9% Rank-1  : 87.9% 来看线上的指标 以这个作为接下来的base
###
##       线上   状态 / 得分 0.48496379618
###
# ---------------------------------------------------------------------- 20200904
## 实验1. base + gemlr*10   mAP: 76.5% Rank-1  : 88.7%             线上   状态 / 得分 0.487
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_moreData_gemlr10 --gpu-devices 0
## 实验2. base + gemlr*10 + bnneck_biasFalse   mAP: 74.7% Rank-1  : 88.5%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_moreData_gemlr10_bias0 --gpu-devices 1
## 实验3. base + gemlr*10 + bnneck_biasFalse + 0.1id 0.9triplet   mAP: 76.8% Rank-1  : 88.8%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_moreData_gemlr10_bias0_wloss --gpu-devices 2
## 实验4. base + gemlr*10 + bnneck_biasFalse + 0.1id 0.9 weighted triplet  mAP: 78.9% Rank-1  : 88.7%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_Weightedtriplet_gpu1_apex_arcface_moreData_gemlr10_bias0_wloss --gpu-devices 3
## 实验5. base + gemlr*10 + bnneck_biasFalse + resnet101  mAP: 62.5% Rank-1  : 78.3%    应该是有bug
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet101_ibn_16x4_90_s1_384x128_ad2_gpu1_apex_arcface_moreData_gemlr10_bias0 --gpu-devices 4
## 实验6. base + gemlr*10 + bnneck_biasFalse + 0.1id 0.9 weighted triplet + batchsize 16*8 mAP: 77.4% Rank-1  : 88.2%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 128 --save-dir log/2020/NAIC/resnet50_ibn_16x8_90_s1_384x128_ad2_gpu2_apex_arcface_moreData_gemlr10_bias0 --gpu-devices 6,7
## 实验7. base + gemlr*10 + bnneck_biasFalse + ranger mAP: 73.7% Rank-1  : 86.8%
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_moreData_gemlr10_bias0_ranger --gpu-devices 5
##           gemlr*10 有用  
##           bnneck_biasFalse 没用 但是还是加上吧
##            0.1id 0.9 weighted triplet 有用 现在最高。     线上竟然降了 唉 ！！！ 线上   状态 / 得分 0.473
##           提升batchsize 没用
#             ranger没用 学习率有问题？
# 由于线上降了，所以最终要的问题是数据集的问题了 先搞下数据集把！！！
# 测试集该怎么划定才能使趋势一致  query，gallery与线上比例一样。
# 增加2019年的数据集 直接在大数据集上训！！！
# ---------------------------------------------------------------------- 20200907
## 实验1. base + gemlr*10   + 2019Data
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/20200907/resnet50_ibn_16x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019Data --gpu-devices 0
## 实验2. base + gemlr*10   + 2019Data 128 16x8
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 128 --save-dir log/20200907/resnet50_ibn_16x8_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019Data --gpu-devices 1,2
## 实验3. base + gemlr*10   + 2019Data 256 32x8     mAP: 15.6% Rank-1  : 19.8%  换了个验证集 特别难 但是现在看和线上趋势一致 就先不换
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200907/resnet50_ibn_32x8_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019Data --gpu-devices 3,4,5,6
#
# 加数据后 线上指标来到了新的阶段  线上   状态 / 得分  0.538 终于上0.5了！！！ 数据真重要啊
# ---------------------------------------------------------------------- 20200908
## 实验1. base + gemlr*10   + 2019Data 256 32x8  resnet101_ibn_a    maxepoch 变成 120   mAP: 14.2% Rank-1  : 19.2%  
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200908/resnet101_ibn_a_32x8_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019Data --gpu-devices 4,5,6,7
## 实验2. base + gemlr*10   + 2019Data 256 32x8  resnet101_ibn_b    maxepoch 变成 120   mAP: 15.8% Rank-1  : 23.6%   线上 状态 / 得分 0.54952196559 
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200908/resnet101_ibn_b_32x8_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019Data --gpu-devices 0,1,2,3
###
#      resnet50_ibn_a --> resnet101_ibn_a 下降了一些。。 奇怪！
##     线上 状态 / 得分 0.54952196559   实验2完成了新高 ！！！ 耶！   
#      所以 resnet101_ibn_b很牛皮
# ---------------------------------------------------------------------- 20200909
## 实验1. base + gemlr*10   + 2019Data 256 32x8  resnet101_ibn_b    maxepoch 变成 90  384x192
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200909/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019Data --gpu-devices 4,5,6,7
## 实验2. base + gemlr*10   + 2019Data 256 64x8  resnet101_ibn_b    maxepoch 变成 90  384x128  biggerBatchsize  64x6
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 384--save-dir log/20200909/resnet101_ibn_b_64x6_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019Data --gpu-devices 0,1,2,3,4,5,6,7
#
#  增大分辨率 384x192有用  增大batchsize有用 错！！！ 现在这两个改变都无法下定论
#  我靠啊 线下涨了很多， 线上又降低了！！！ 唉 线下测试集的设定太重要了， 没设定好就容易出现涨跌不一致的情况，就无法分辨模型的好坏了！！！
#                             线下测试集 重中之重。
#    数据！！！ 复赛数据包括初赛。 需要重新操作一下
# ---------------------------------------------------------------------- 20200910
## 实验1. base + gemlr*10   + 2019Data 256 32x8  resnet101_ibn_b    maxepoch 变成 90  384x192  数据踢出初赛
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200910/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew --gpu-devices 0,1,2,3
## 实验2. base + gemlr*10   + 2019Data 384 64x6  resnet101_ibn_b    maxepoch 变成 90  384x128  数据踢出初赛
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 384 --save-dir log/20200910/resnet101_ibn_b_64x6_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew --gpu-devices 0,1,2,3,4,5,6,7
## 实验3. base + gemlr*10   + 2019Data 256 32x8  resnet101_ibn_b    maxepoch 变成 90  384x192  数据踢出初赛 0.1id 0.9triplet 状态 / 得分 0.54767494650 
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200910/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss --gpu-devices 4,5,6,7
# ---------------------------------------------------------------------- 20200911
#     线上线下不一致 对我打击很大。 最近两天都没有进展 很恼火。。。
#     听说大佬们都是用单卡在训练，这就挺有意思，那我这边增加batchsize是不是不咋对。
#     并且单卡可以做的实验会很多。 所以我想也改成单卡， 但是有个问题是咱们这个单卡显存大小还是有点不足，所以有时也可以用双卡。 现在先把syncBN写出来
###
#       
## todo  384x192  dataAug  efficientnet biggerBatchsize syncBatchNorm 
## 实验1. base + gemlr*10   + 2019Data 256 32x8  resnet101_ibn_b    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  16.9 23.9  状态 / 得分 0.55450027953 
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200911/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN --gpu-devices 0,1,2,3

## 实验1. base + gemlr*10   + 2019Data 64 16x4  resnet101_ibn_b    maxepoch 变成 90  384x128  数据踢出初赛 0.33id 1triplet  8.8  12.1
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/20200911/resnet101_ibn_b_64x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss --gpu-devices 0
## 实验2. base + gemlr*10   + 2019Data 64 16x4  resnet101_ibn_b    maxepoch 变成 90  384x128  数据踢出初赛 0.33id 1triplet + cj0.5 7.4 10.5
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/20200911/resnet101_ibn_b_64x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_cj0.5 --gpu-devices 1
## 实验3. base + gemlr*10   + 2019Data 64 16x4  resnet101_ibn_b    maxepoch 变成 90  384x128 数据踢出初赛 0.33id 1triplet + dataAug 11.3 17.3
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/20200911/resnet101_ibn_b_64x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_dataAug --gpu-devices 2
## 实验4. base + gemlr*10   + 2019Data 64 16x4  resnet101_ibn_b    maxepoch 变成 90  384x128 数据踢出初赛 0.1id 0.9triplet 8.6 11.9
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/20200911/resnet101_ibn_b_64x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss0.1 --gpu-devices 3
## 实验5. base + gemlr*10   + 2019Data 64 16x4  resnet101_ibn_b    maxepoch 变成 90  384x128 数据踢出初赛 0.33id 1 weighted triplet   8.4 11.9  
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/20200911/resnet101_ibn_b_64x4_90_s1_384x128_ad2_Weightedtriplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss --gpu-devices 4
## 实验6. base + gemlr*10   + 2019Data 64 16x4  resnet101_ibn_b    maxepoch 变成 90  384x128  数据踢出初赛 0.33id 1triplet + noMaigin 6.9 10.9
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/20200911/resnet101_ibn_b_64x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_noMargin --gpu-devices 5
## 实验7. base + gemlr*10   + 2019Data 64 16x4  resnet101_ibn_b    maxepoch 变成 90  384x128  数据踢出初赛 0.33id 1triplet  - labelsmooth  9.6 12.4
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/20200911/resnet101_ibn_b_64x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_noSmooth --gpu-devices 6
## 实验8. base + gemlr*10   + 2019Data 64 16x4  resnet101_ibn_b    maxepoch 变成 90  384x128  数据踢出初赛 0.33id 1triplet  load_weights 8.9 12.9
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/20200911/resnet101_ibn_b_64x4_90_s1_384x128_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_lweight --gpu-devices 7

# ---------------------------------------------------------------------- 20200914
#           batchsize 真的影响很大啊 几乎不是一个级别的 那咋用单卡啊 真是奇奇怪怪
##          dataAug那一套有点用 no label smooth 和 load_weight似乎有点用
##          triplet margin 和 weightTriplet 还有权重的调整没用
##          同步BN线上涨粉了

## 实验1. base + gemlr*10   + 2019Data 256 32x8  resnet101_ibn_b    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  dataAug       19.1 28.9   0.558
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200914/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug --gpu-devices 0,1,2,3
## 实验2. base + gemlr*10   + 2019Data 256 32x8  resnet101_ibn_b    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  dataAug noSmooth    19.0 29.0  0.57081757868   线下不差线上差这么多。。。
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200914/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth --gpu-devices 4,5,6,7
## 实验3. base + gemlr*10   + 2019Data 384 64x6  resnet101_ibn_b    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  dataAug noSmooth bigger batchsize  21.4 32.4   0.56839869477   比上面低  555555555555555555
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200914/resnet101_ibn_b_64x6_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug --gpu-devices 0,1,2,3,4,5,6,7
#    nolabel smooth 有提升
#      有出现线上线下不匹配了  暂时不想管了。。 到时候想搞数据的时候再来看吧 现在线上可以达到 状态 / 得分 0.57081757868 
# ---------------------------------------------------------------------- 20200915
## 实验1. base + gemlr*10   + 2019Data 384 64x6  resnet101_ibn_b    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  dataAug noSmooth 
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 384 --save-dir log/20200914/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug --gpu-devices 0,1,2,3,4,5,6,7
## 实验2. base + gemlr*10   + 2019Data 128 32x8 efficientnet-b4    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  dataAug noSmooth 
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 128 --save-dir log/20200915/efficientb4_32x4_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth --gpu-devices 0,1,2,3
## 实验3. base + gemlr*10   + 2019Data 384 32x8  efficientnet-b0    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  dataAug noSmooth 
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200915/efficientb0_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth --gpu-devices 4,5,6,7
#  训练efficientnet出现了些问题， 消耗显存太大了 所以最近的实验都有点小问题。
#
# ---------------------------------------------------------------------- 20200917
## 实验1. base + gemlr*10   + 2019Data 256 32x8  resnet101_ibn_b    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  dataAug noSmooth 
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200917/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_RP --gpu-devices 0,1,2,3
## 实验2. base + gemlr*10   + 2019Data 256 32x8  resnet101_ibn_b    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  dataAug noSmooth 
python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 256 --save-dir log/20200917/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_std --gpu-devices 4,5,6,7



## 实验1. base + gemlr*10   + 2019Data 128 16x4 efficientnet-b4    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  dataAug noSmooth 
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 64 --save-dir log/20200917/efficientb4_16x4_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth --gpu-devices 0,1,2,3
## 实验2. base + gemlr*10   + 2019Data 128 32x8 efficientnet-b4    maxepoch 变成 90  384x192  数据踢出初赛 0.33id 1 triplet  dataAug noSmooth 
#python train_imgreid_xent.py --config_type resnet50_bagOfTricks  --dataset NAIC_2020  --train-batch 128 --save-dir log/20200917/efficientb4_16x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_augGrad --gpu-devices 4,5,6,7



###  aqe 可以涨分！！！！。
#scp -r xuzihao@192.168.9.251:/data/xuzihao/NAIC/ReID/code/train.sh ./

#
### model efficient-net  ibn-b 101
### center loss
## ranger
## rerank args

