import json
import os
import glob
import shutil
from tqdm import tqdm
from collections import defaultdict
# kp_path = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/test/1/_json'
# pic_path = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/test/1'
# test_path = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/test/badGuy'

main_path = '/data/xuzihao/NAIC/ReID/code'

notRed_txt = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/list_train_img_notRed_allData.txt'
wash_txt = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/list_train_img_notRed_allData_wash.txt'
# if not os.path.exists(test_path):
#     os.mkdir(test_path)

#### wash data
# id_dict = []
# with open(notRed_txt, 'r') as fr, open(wash_txt, 'w') as fw:
#     lines = fr.readlines()
#     count = 0
#     for line in tqdm(lines):
#         count += 1
#         img_path = line.split(' ')[0]
#         img_id = line.split(' ')[1].strip()
#         json_path = os.path.join(main_path, os.path.dirname(img_path), '_json', img_path.split('/')[-1].replace('.png', '.json'))
#         with open(json_path, 'r') as fj:
#             kp = json.load(fj) 
#             body_score = 0
#             for i in range(17):
#                 body_score += kp[0]['keypoints']['__ndarray__'][i][2]
            
#             if body_score > 3 and img_id not in id_dict:
#                 temp_line = line
#                 id_dict.append(img_id)          
#             elif body_score > 3 and img_id in id_dict:
#                 if temp_line != '':
#                     fw.write(temp_line)
#                     temp_line = ''
#                 fw.write(line)

#### delete id more than 50
main_path = '/data/xuzihao/NAIC/ReID/code'

notRed_txt = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/list_train_img_notRed_allData_wash.txt'
wash_txt = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/list_train_img_notRed_allData_wash_delete50.txt'

id_dict = defaultdict(int)
with open(notRed_txt, 'r') as fr, open(wash_txt, 'w') as fw:
    lines = fr.readlines()
    for line in lines:
        img_id = line.split(' ')[1].strip()
        id_dict[img_id] += 1

    for line in tqdm(lines):
        img_id = line.split(' ')[1].strip()
        if id_dict[img_id] <= 50:
            fw.write(line)







# json_list = glob.glob(kp_path + '/*.json')
# score_dict = {}
# for each in json_list:
#     with open(each, 'r') as fr:
#         kp = json.load(fr)
#        # print(each)
#         body_score = 0
#         for i in range(17):
#             body_score += kp[0]['keypoints']['__ndarray__'][i][2]
#         score_dict[each.split('/')[-1]] = body_score
#         if body_score < 3:
#             print(each)
#             print(body_score)
#             shutil.copy(os.path.join(pic_path, each.split('/')[-1].replace('.json', '.png')), os.path.join(test_path, each.split('/')[-1].replace('.json', '.png')))
#print(sorted(score_dict.items()))
