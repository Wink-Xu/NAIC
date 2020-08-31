import os
from collections import defaultdict

datapath = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/train/label.txt'

picNum = defaultdict(int)

with open(datapath, 'r') as fr:
    lines = fr.readlines()
    for each in lines:
        picNum[each.split(':')[-1].strip()] += 1

print(picNum)


lessNum = 0
for i in picNum.keys():
    if picNum[i] <= 2:
        lessNum += 1
print(lessNum)


