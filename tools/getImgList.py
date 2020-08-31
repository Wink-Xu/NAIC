# -*- coding: utf-8 -*-


import os
import shutil
from collections import defaultdict


### Get train and val img list 

datapath = '../../data/NAIC_2020/train/label.txt'
mask_dict = []
picNum = defaultdict(int)
with open(datapath, 'r') as fr:
    lines = fr.readlines()
    for each in lines:
        picNum[each.split(':')[-1].strip()] += 1

for i in picNum.keys():
    if picNum[i] >= 3 and picNum[i] <= 100:
        mask_dict.append(i)
mask_dict = mask_dict[:5000]

main_dir = '../../data/NAIC_2020/train'
with open(os.path.join(main_dir, 'list_train_img_all_no123_5000.txt'), 'w') as ft, open(os.path.join(main_dir, 'list_query_img.txt'), 'w') as fq, \
                                                                  open(os.path.join(main_dir, 'list_gallery_img.txt'), 'w') as fg:
    query_dir = main_dir
    with open(os.path.join(query_dir, 'label.txt'), 'r') as fr:
        lines = fr.readlines()
        dict_eval = defaultdict(list)
        for index, i in enumerate(lines):
            imgpath = os.path.join('images', i.split(':')[0])
            imgid = i.split(':')[1]
            if imgid.strip() in mask_dict:
                ft.write(imgpath + ' ' + imgid)
            if index > 67240:
                imgid = i.split(':')[1]
                dict_eval[imgid].append( os.path.join('images', i.split(':')[0]) )
        for each in dict_eval.keys():
            if len(dict_eval[each]) > 2:
                fq.write(dict_eval[each][0] + ' ' + each)
                for j in range(1, len(dict_eval[each])):
                    fg.write(dict_eval[each][j] + ' ' + each)
            else:
                for j in range(len(dict_eval[each])):
                    fg.write(dict_eval[each][j] + ' ' + each)

#### Get test img list

# main_dir = '../../data/NAIC_2020/image_A'
# with open(os.path.join(main_dir, 'list_query_img.txt'), 'w') as fq, open(os.path.join(main_dir, 'list_gallery_img.txt'), 'w') as fg:
#     query_dir = os.path.join(main_dir, 'query')
#     for index, i in enumerate(os.listdir(query_dir)):
#         if i.endswith('.png'):
#             imgpath = os.path.join('query', i)
#             fq.write(imgpath + ' ' + str(index))
#             fq.write('\n')
#     gallery_dir = os.path.join(main_dir, 'gallery')
#     for index, i in enumerate(os.listdir(gallery_dir)):
#         if i.endswith('.png'):
#             imgpath = os.path.join('gallery', i)
#             fg.write(imgpath + ' ' + str(index))
#             fg.write('\n')

