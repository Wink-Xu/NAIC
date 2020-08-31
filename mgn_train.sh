#!/usr/bin/env bash 
export PATH="/home1/chaizh/anaconda3/bin:$PATH" 
# change evaluate
# my
python train_imgreid_xent.py --root ../data -d VeRi776 -a resnet50v3a --load-weight /data/xuzihao/ReID/code/log/2020/vehicleReID/VeRi776/resnet50_16x4_0.00035_adam_250_ls_batch_clsbalance_gpu0_256x128_doPad/best_model.pth.tar  --evaluate --vis-ranked-res --save-dir log/2020/vehicleReID/VeRi776/show_Result/res1 --gpu-devices 1 --height=256 --width=128
#czh
#python train_imgreid_xent.py --root ../data -d market1501 -a mgresnet50v3a --load-weight /data/xuzihao/ReID/code/best_model.pth.tar --evaluate --vis-ranked-res --save-dir log/MGNetV3/market1501/i__market1501__ --gpu-devices 4,5,6,7 --height=384 --width=128

#python train_imgreid_xent_htri.py --root ../data -d vionData -a mgresnet50v3x --load-weight /data/xuzihao/ReID/code/log/mgresnet50v3x-xent-htri-msmt17_localfeat_480/best_model.pth.tar --evaluate --vis-ranked-res --save-dir log/MGNetV3/vionData --gpu-devices 4,5,6,7 --height=384 --width=128
#python train_imgreid_xent_htri.py --root ../data -d vionData -a mgresnet50v3x --load-weight /data/xuzihao/ReID/code/log/mgresnet50v3x-xent-htri-msmt17_localfeat_480/best_model.pth.tar --evaluate --vis-ranked-res --save-dir log/MGNetV3/vionData --gpu-devices 4,5,6,7 --height=384 --width=128

# triplet
#python train_imgreid_xent_htri.py --root ../data -d msmt17 -a mgresnet50v3x --label-smooth --eval-step 10 \
#	--optim adam --lr 0.0003 --max-epoch 100 --stepsize 20 60 --train-batch 64 --test-batch 100 \
#	--save-dir log/mgresnet50v3x-xent-htri-msmt17_localfeat_480 --gpu-devices 0,1,2,3 

#python train_imgreid_xent.py --root ../data -d wuYueData -a mgresnet50v3a --class-balance --eval-step 40 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 400 --stepsize 110 200 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/MGNetV3/wuYue_V1_align/nmgresnet50v3a-all-x-wuYue_v1 --height=384 --width=128 \
#        --gpu-devices 4,5,6,7 --warmup 1

#python train_imgreid_xent.py --root ../data -d msmt17 -a mgresnet50v3x --class-balance --eval-step 40 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 400 --stepsize 110 200 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/MGNetV3/MSMT17_V1_align/nmgresnet50v3x-all-x-msmt17_v1 --height=384 --width=128 \
#        --gpu-devices 0,1,2,3 --warmup 1

#python train_imgreid_xent_htri.py --root ../data -d msmt17 -a mgresnet50v3a --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.01 --max-epoch 60 --stepsize 20 40 --train-batch 32 --test-batch 100 \
#	--save-dir log/MGNetV3_htri/MSMT17_V1_align/mgresnet50v3a-all-x-msmt17_v1_allData3_0.01_ --height=384 --width=128 \
#         --gpu-devices 4,5,6,7
#python train_imgreid_xent.py -d msmt17 -a mgseresnet50v3a --class-balance --eval-step 30 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 200 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/mgseresnet50v3a-all-x-msmt17L-RE --height=384 --width=128 --gpu-devices 6,7,8,9
#python train_imgreid_xent.py -d msmt17 -a mgseresnet50v3a --class-balance --eval-step 40 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 400 --stepsize 110 200 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/MGSENetV3/msmt17AL/mgseresnet50v3a-all-x-msmt17IL --height=384 --width=128 --gpu-devices 6,7,8,9

# ------------------------------------  market1501
#python train_imgreid_xent.py --root ../data -d market1501 -a mgresnet50v3x_1 --class-balance --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 170 280 --train-batch 32 --test-batch 100 \
#	--save-dir log/market1501/mgresnet50v3x_1-base_x_32 --height=384 --width=128 --gpu-devices 4,5,6,7


# triplet
#python train_imgreid_xent_htri.py --root ../data -d market1501 -a mgresnet50v3x --label-smooth --eval-step 10 \
#	--optim adam --lr 0.0003 --max-epoch 100 --stepsize 20 60 --train-batch 64 --test-batch 100 \
#	--save-dir log/market1501/mgresnet50v3x-base_htri_RE_x --gpu-devices 4,5,6,7 

#python train_imgreid_xent.py -d msmt17 -a mgseresnet50v3a --class-balance --eval-step 30 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 200 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/mgseresnet50v3a-all-x-msmt17L-RE --height=384 --width=128 --gpu-devices 6,7,8,9

# MGSENETV3
#python train_imgreid_xent.py -d msmt17 -a mgseresnet50v3a --class-balance --eval-step 30 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 200 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/MGSENetV3/mgseresnet50v3a-all-x-msmt17IL-ag --height=384 --width=128 --gpu-devices 2,3,4,5
#python train_imgreid_xent.py -d dukemtmcreid -a mgseresnet50v3a --class-balance --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 200 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/MGSENetV3/mgseresnet50v3a-all-x-dukeL --height=384 --width=128 --gpu-devices 6,7,8,9
#python train_imgreid_xent.py -d cuhk03 -a mgseresnet50v3a --class-balance --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 200 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/MGSENetV3/mgseresnet50v3a-all-x-cuhk03L --height=384 --width=128 --gpu-devices 2,3,4,5

#python train_imgreid_xent.py -d market1501 -a mgseresnet50v3a --class-balance=True --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 170 280 --train-batch 128 --test-batch 100 \
#	--resume log/MGSENetV3/mgseresnet50v3a-all-x-mkt1501L/best_model.pth.tar \
#	--save-dir log/MGSENetV3/mgseresnet50v3a-all-x3-mkt1501L --height=384 --width=128 --gpu-devices 2,3,4,5,6,7,8,9
#python train_imgreid_xent.py -d market1501 -a mgseresnet50v3b --class-balance=True --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 170 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/MGSENetV3/mgseresnet50v3b-all-x-mkt1501L --height=384 --width=128 --gpu-devices 6,7,8,9
#python train_imgreid_xent.py -d market1501 -a mgseresnet50v3a --class-balance=True --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 170 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/mgseresnet50v3a-am-all-x-mkt1501L --height=384 --width=128 --gpu-devices 2,3,4,5
#python train_imgreid_xent.py -d market1501 -a mgseresnet50v3 --class-balance=True --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 170 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/mgseresnet50v3-all-x-mkt1501L --height=384 --width=128 --gpu-devices 6,7,8,9


# MGNETV3
#python train_imgreid_xent_htri.py -d market1501 -a mgresnet50v3 --margin=1.2 --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.1 --lambda-htri=0.1 --max-epoch 80 --stepsize 20 50 70 --train-batch 64 --test-batch 100 \
#	--save-dir log/mgresnet50v3-alld1-xh-mkt1501L --height=384 --width=128 --gpu-devices 0,1
#python train_imgreid_xent.py -d market1501 -a mgresnet50v3 --class-balance=True --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 100 180 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/mgresnet50v3-alla-x-mkt1501L --height=384 --width=128 --gpu-devices 6,7
#python train_imgreid_xent.py -d market1501 -a mgresnet50v3a --class-balance=True --eval-step 10 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 170 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/mgresnet50v3a-all-x-mkt1501L --height=384 --width=128 --gpu-devices 2,3,4,5

# triplet
#python train_imgreid_xent_htri.py --root ../data -d msmt17 -a resnet50v3 --label-smooth --eval-step 10 \
#	--optim adam --lr 0.0003 --max-epoch 100 --stepsize 20 60 --train-batch 64 --test-batch 100 \
#	--save-dir log/resnet50v3-xent-htri-msmt17 --gpu-devices 2,3 



#python train_imgreid_xent.py -d msmt17 -a mgresnet50v3a --class-balance=True --eval-step 20 --workers=16 \
#	--optim sgd --lr 0.1 --max-epoch 300 --stepsize 110 170 280 --train-batch 64 --test-batch 100 \
#	--save-dir log/MGNetV3/mgresnet50v3a-all-x-msmt17L --height=384 --width=128 --gpu-devices 0,1,8,9


# MGNETV2
#python train_imgreid_xent_htri.py -d market1501 -a mgresnet50v2 --label-smooth --eval-step 10 \
#	--optim adam --lr 0.0003 --max-epoch 60 --stepsize 20 40 --train-batch 32 --test-batch 100 \
#	--save-dir log/mgresnet50v2-b2-xh-mkt1501 --gpu-devices 4,5 
#python train_imgreid_xent_htri.py -d market1501 -a mgresnet50v2 --label-smooth --eval-step 10 \
#	--optim adam --lr 0.0003 --max-epoch 120 --stepsize 20 40 --train-batch 32 --test-batch 100 \
#	--save-dir log/mgresnet50v2-b2-xh-mkt1501L --height=384 --width=128 --gpu-devices 6,7
#python train_imgreid_xent_htri.py -d market1501 -a mgresnet50v2 --label-smooth --eval-step 10 --lambda-xent=1 \
#	--optim adam --lr 0.0003 --max-epoch 120 --stepsize 20 40 --train-batch 32 --test-batch 100 \
#	--save-dir log/mgresnet50v2-b2z-xh-mkt1501L --height=384 --width=128 --gpu-devices 8,9
#python train_imgreid_xent_htri.py -d market1501 -a mgresnet50v2 --label-smooth --eval-step 10 \
#	--optim adam --lr 0.0003 --max-epoch 120 --stepsize 20 40 --train-batch 32 --test-batch 100 \
#	--save-dir log/mgresnet50v2-all-xh-mkt1501 --gpu-devices 8,9 


