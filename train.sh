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
python train_imgreid_xent.py --config_type mgresnet50v3a  --dataset NAIC_2020  --train-batch 64 --save-dir log/2020/NAIC/mgresnet50v3a_bot_16x4_120_noS_384x128_ad2_aug --gpu-devices 0,1
