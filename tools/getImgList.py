# -*- coding: utf-8 -*-


import os
import shutil
from collections import defaultdict



############  get all data 


# datapath = '../../data/NAIC_2020/rematch/train/label.txt'
# mask_dict = []
# picNum = defaultdict(int)
# with open(datapath, 'r') as fr:
#     lines = fr.readlines()
#     for each in lines:
#         picNum[each.split(':')[-1].strip()] += 1


# main_dir = '../../data/NAIC_2020/rematch/train'
# with open(os.path.join(main_dir, 'list_train_img.txt'), 'w') as ft:
#     query_dir = main_dir
#     with open(os.path.join(query_dir, 'label.txt'), 'r') as fr:
#         lines = fr.readlines()
#         for index, i in enumerate(lines):
#             imgid = i.split(':')[1]
#             if picNum[imgid.strip()] >= 2:    
#                 imgpath = os.path.join('../data/NAIC_2020/rematch/train/images', i.split(':')[0])
#                 ft.write(imgpath + ' ' + imgid)


# picNum = defaultdict(int)
# fuSaipath = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/extraTrain/REID2019_fusai/train_list.txt'
# with open(fuSaipath, 'r') as fr:
#     lines = fr.readlines()
#     for each in lines:
#         picNum[each.split(' ')[-1].strip()] += 1

# with open(os.path.join('../../data/NAIC_2020/train', 'list_train_img_allImg.txt'), 'a') as ft:
#     with open(os.path.join(fuSaipath), 'r') as fr:
#         lines = fr.readlines()
#         for index, i in enumerate(lines):
#             imgid = i.split(' ')[-1]
#             if picNum[imgid.strip()] >= 2:     
#                 imgpath = os.path.join('../data/NAIC_2020/extraTrain/REID2019_fusai/fusai_2019_1', i.split(' ')[0][6:])
#                 ft.write(imgpath + ' ' + imgid.strip() + '_fs')
#                 ft.write('\n')

###         add no label data

# noLabelImgPath = '../../data/NAIC_2020/train/noLabelImg_clusters'
# with open(os.path.join('../../data/NAIC_2020/train', 'list_train_img_allImg_withNoLabelData.txt'), 'a') as ft:
#     for each in os.listdir(noLabelImgPath):
#         temp_dir = os.path.join(noLabelImgPath, each)
#         picNum = len(os.listdir(temp_dir))
#         if picNum >= 4:
#             for imgname in os.listdir(temp_dir):
#                 if imgname.endswith('.png'):
#                     imgid = each                
#                     imgpath = os.path.join('../data/NAIC_2020/train/noLabelImg_clusters', each, imgname)
#                     ft.write(imgpath + ' ' + imgid + '_noLabel')
#                     ft.write('\n')



# ### Get train and val img list 

# datapath = '../../data/NAIC_2020/train/label.txt'
# mask_dict = []
# picNum = defaultdict(int)
# with open(datapath, 'r') as fr:
#     lines = fr.readlines()
#     for each in lines:
#         picNum[each.split(':')[-1].strip()] += 1

# valData_query_ids = []
# valData_gallery_ids = []
# query_ids = 0
# gallery_nums = 0
# for i, num in picNum.items():
#     if picNum[i] == 1:
#         valData_gallery_ids.append(i)
#         gallery_nums += 1
#     elif picNum[i] >= 3 and query_ids <= 1500:
#         valData_query_ids.append(i)
#         query_ids += 1
#         gallery_nums += num - 1
#     elif gallery_nums <= 21000 and num > 20:
#         valData_gallery_ids.append(i)
#         gallery_nums += num


# main_dir = '../../data/NAIC_2020/train'
# with open(os.path.join(main_dir, 'list_train_img_all_ratio.txt'), 'w') as ft, open(os.path.join(main_dir, 'list_query_img_ratio.txt'), 'w') as fq, \
#                                                                   open(os.path.join(main_dir, 'list_gallery_img_ratio.txt'), 'w') as fg:
#     query_dir = main_dir
#     with open(os.path.join(query_dir, 'label.txt'), 'r') as fr:
#         lines = fr.readlines()
#         dict_eval = defaultdict(list)
#         for index, i in enumerate(lines):
#             imgid = i.split(':')[1]
#             if imgid.strip() in valData_query_ids:     
#                 dict_eval[imgid].append( os.path.join('../data/NAIC_2020/train/images', i.split(':')[0]) )
#             elif imgid.strip() in valData_gallery_ids:   
#                 imgpath = os.path.join('../data/NAIC_2020/train/images', i.split(':')[0])
#                 fg.write(imgpath + ' ' + imgid)
#             else:
#                 imgpath = os.path.join('../data/NAIC_2020/train/images', i.split(':')[0])
#                 ft.write(imgpath + ' ' + imgid)
#         for each in dict_eval.keys():
#             if len(dict_eval[each]) > 2:
#                 fq.write(dict_eval[each][0] + ' ' + each)
#                 for j in range(1, len(dict_eval[each])):
#                     fg.write(dict_eval[each][j] + ' ' + each)
#             else:
#                 for j in range(len(dict_eval[each])):
#                     fg.write(dict_eval[each][j] + ' ' + each)


# # # #### Get extra train img list
# # chuSaipath = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/extraTrain/REID2019_chusai/train_list.txt'

# # picNum = defaultdict(int)
# # with open(chuSaipath, 'r') as fr:
# #     lines = fr.readlines()
# #     for each in lines:
# #         picNum[each.split(' ')[-1].strip()] += 1

# # main_dir = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/extraTrain/REID2019_chusai/chusai_2019'
# # with open(os.path.join('../../data/NAIC_2020/train', 'list_train_img_all_ratio.txt'), 'a') as ft:
# #     query_dir = main_dir
# #     with open(os.path.join(chuSaipath), 'r') as fr:
# #         lines = fr.readlines()
# #         for index, i in enumerate(lines):
# #             imgid = i.split(' ')[-1]
# #             if picNum[imgid.strip()] >= 2:     
# #                 imgpath = os.path.join('../data/NAIC_2020/extraTrain/REID2019_chusai/chusai_2019', i.split(' ')[0][6:])
# #                 ft.write(imgpath + ' ' + imgid.strip() + '_cs')
# #                 ft.write('\n')

# picNum = defaultdict(int)
# fuSaipath = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/extraTrain/REID2019_fusai/train_list.txt'
# with open(fuSaipath, 'r') as fr:
#     lines = fr.readlines()
#     for each in lines:
#         picNum[each.split(' ')[-1].strip()] += 1

# with open(os.path.join('../../data/NAIC_2020/train', 'list_train_img_all_ratio.txt'), 'a') as ft:
#     with open(os.path.join(fuSaipath), 'r') as fr:
#         lines = fr.readlines()
#         for index, i in enumerate(lines):
#             imgid = i.split(' ')[-1]
#             if picNum[imgid.strip()] >= 2:     
#                 imgpath = os.path.join('../data/NAIC_2020/extraTrain/REID2019_fusai/fusai_2019_1', i.split(' ')[0][6:])
#                 ft.write(imgpath + ' ' + imgid.strip() + '_fs')
#                 ft.write('\n')

# #### Get test img list

main_dir = '../../data/NAIC_2020/rematch/image_B_v1.1'
with open(os.path.join(main_dir, 'list_query_img_rematch_B_normal.txt'), 'w') as fq, open(os.path.join(main_dir, 'list_gallery_img_rematch_B_normal.txt'), 'w') as fg:
    query_dir = os.path.join(main_dir, 'query_normal')
    for index, i in enumerate(os.listdir(query_dir)):
        if i.endswith('.png'):
            imgpath = os.path.join('query_normal', i)
            fq.write(imgpath + ' ' + str(index))
            fq.write('\n')
    gallery_dir = os.path.join(main_dir, 'gallery_normal')
    for index, i in enumerate(os.listdir(gallery_dir)):
        if i.endswith('.png'):
            imgpath = os.path.join('gallery_normal', i)
            fg.write(imgpath + ' ' + str(index))
            fg.write('\n')

