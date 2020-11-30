import os
from collections import defaultdict

mainpath = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/train/list_train_img_notRed.txt'

trainpath = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/train/list_train_img_notRed_ttt.txt'
querypath = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/train/list_query_img_notRed.txt'
gallerypath = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/train/list_gallery_img_notRed.txt'

img_dict = defaultdict(int)
with open(mainpath, 'r') as fr:
    lines = fr.readlines()
    for each in lines:
        pid = each.split(' ')[-1]
        img_dict[pid] += 1

id_list = []

for key in img_dict.keys():
    if img_dict[key] >= 2 and img_dict[key] <= 4:
        id_list.append(key)

print(len(id_list))

id_list = id_list[::14]
print(len(id_list))
import pdb;pdb.set_trace()
# #  not red
with open(mainpath, 'r') as fr, open(trainpath, 'w') as fr1, open(querypath, 'w') as fq, open(gallerypath, 'w') as fg:
    lines = fr.readlines()
    flag_list = []
    for each in lines:
        pid = each.split(' ')[-1]
        if pid in id_list:
            if pid not in flag_list:
                fq.write(each)
                flag_list.append(pid)
            else:
                fg.write(each)
        else:
            fr1.write(each)



#  red
# with open(mainpath, 'r') as fr, open(trainpath, 'w') as fr1, open(querypath, 'w') as fq, open(gallerypath, 'w') as fg:
#     lines = fr.readlines()
#     flag_list = []
#     for each in lines:
#         pid = each.split(' ')[-1]
#         if pid in id_list:
#             if pid not in flag_list:
#                 fq.write(each)
#                 flag_list.append(pid)
#             else:
#                 fg.write(each)
#         else:
#             fr1.write(each)

    