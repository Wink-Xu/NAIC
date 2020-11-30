import json

json_1 = '/data/xuzihao/NAIC/ReID/code/log/20201029/resnet101_ibn_b_32x8_90_s1_384x192_rematch_base_red_B/test_rematch.json'
with open(json_1, 'r') as fp:
    json_one = json.load(fp)
json_2 = '/data/xuzihao/NAIC/ReID/code/result_for_ensemble_C/temp.json'
with open(json_2, 'r') as fp1:
    json_two = json.load(fp1)


last_json = json_one.copy()
last_json.update(json_two)


last_json_path = '/data/xuzihao/NAIC/ReID/code/result_for_ensemble_C/rematch_B.json'
with open(last_json_path, 'w') as fw:
    json.dump(last_json, fw)

# import shutil
# import os
# imglist = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/train/list_train_img_notRed.txt'
# imgpath = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/train/images'
# dstpath = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/train/notRed'
# if not os.path.exists(dstpath):
#     os.mkdir(dstpath)
# cnt = 0
# with open(imglist, 'r') as fr:
#     lines = fr.readlines()
#     for each in lines:
#         imgname = each.split(' ')[0].split('/')[-1]
#         shutil.copy(os.path.join(imgpath, imgname), os.path.join(dstpath, imgname))
#         cnt += 1

