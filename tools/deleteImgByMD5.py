import hashlib
import os
import json
from tqdm import tqdm

rematch_train_path = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/train/images'
chushai_train_path = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/train/images'


md5_dict = {}
for i in tqdm(os.listdir(rematch_train_path)):
    if i.endswith('.png'):
        temp_dir = os.path.join(rematch_train_path, i)
        file = open(temp_dir, 'rb')
        md5 = hashlib.md5(file.read()).hexdigest()
        
        md5_dict[md5] = i

result_json = {}
for j in tqdm(os.listdir(chushai_train_path)):
    if j.endswith('.png'):
        temp_dir = os.path.join(chushai_train_path, j)
        file = open(temp_dir, 'rb')
        md5 = hashlib.md5(file.read()).hexdigest()
        
        if md5 in md5_dict:
            result_json[j] = 1
json_result_path = './chusai_in_rematch.json'
with open(json_result_path, 'w') as fw:
    json.dump(result_json, fw)


    



file = open('/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/train/images/00207382.png','rb')
md5 = hashlib.md5(file.read()).hexdigest()
print(md5)