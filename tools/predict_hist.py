import cv2
import glob
import mmcv
import numpy as np 
import pandas as pd
import shutil
import os
from tqdm import tqdm



def compute(img):

    per_image_Bmean = np.mean(img[:, :, 0])
    per_image_Gmean = np.mean(img[:, :, 1])
    per_image_Rmean = np.mean(img[:, :, 2])
    if per_image_Bmean > 65 and per_image_Gmean > 65 and per_image_Rmean > 65:
        return 0   # green
    else:
        return 1   # red


def simple_hist_predictor(image,channel=2,thres=100): #BGR; by the last channel
    hist = cv2.calcHist([image], [channel], None, [256], [0, 256]) #绘制各个通道的直方图
    if hist[0]>thres:
        return 0   #green
    else:
        return 1   # red

if __name__ == "__main__":
    if 0:
        root_dir = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/extraTrain/REID2019_fusai/'
        df_train = pd.read_csv(root_dir+'train_list.txt',sep=' ',header=None,names=['filename','identity'])
        #df_train['filename'] = df_train['filename'].apply(lambda x: x.split('/')[1]) 
        df_train['filename'] = df_train['filename'].apply(lambda x: root_dir + 'fusai_2019_1/' + x.split('/')[-1]) 
        # df_train_partial = df_train.head(100)
        #df_train_partial = df_train.head(10000)
        df_train_partial = df_train

        def func(fname):
            img = cv2.imread(fname)
            return simple_hist_predictor(img,channel=2,thres=100)
        labels = mmcv.track_parallel_progress(func, df_train_partial['filename'], 8)
        import pdb;pdb.set_trace()
        df_train_partial['label'] = labels
        print("==> distribution for hist class")
        print(df_train_partial['label'].value_counts())
        df_id_cnt = df_train_partial.groupby(df_train_partial['identity']).apply(lambda x: len(np.unique(x['label']))).reset_index().rename(columns={0:'cnt'})
        print("==> distribution for the number of each identity")
        print(df_id_cnt['cnt'].value_counts())
        print("total identity:",len(set(df_train_partial['identity'].to_list())))
        df_train_partial.to_csv('2019_fu.csv')
     #   import pdb;pdb.set_trace()
    if 0:
        root_dir = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/unlabel/noLabelImg_dbscan_0.6_2_1000'
        notRed_txt_path = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/unlabel/noLabelImg_notRed.txt'
        red_txt_path = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/unlabel/noLabelImg_red.txt'
        txt_path = '../data/NAIC_2020/rematch/unlabel/noLabelImg_dbscan_0.6_2_1000'
        img_list = glob.glob(root_dir+'/*/*.png')

        def func(fname):
            img = cv2.imread(fname)
            return simple_hist_predictor(img,channel=2,thres=100)
        labels = mmcv.track_parallel_progress(func, img_list, 8)

        with open(notRed_txt_path, 'w') as fw1, open(red_txt_path, 'w') as fw2:
            for i in tqdm(range(len(labels))):
                id = img_list[i].split('/')[-2]
                img_name = img_list[i].split('/')[-1]
                if labels[i] == 0:
                    fw2.write(os.path.join(txt_path, id, img_name) + ' ' + 'unlabel_' + id)
                    fw2.write('\n')
                else:
                    fw1.write(os.path.join(txt_path, id, img_name) + ' ' + 'unlabel_' + id)
                    fw1.write('\n')

    if 0:
        img_dir = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/unlabel/noLabelImg_dbscan_0.6_2_1000'

        txt_path = '../data/NAIC_2020/rematch/unlabel/noLabelImg_dbscan_0.6_2_1000'
        for idx in os.listdir(img_dir):
            temp_dir = os.path.join(img_dir, idx)
            for img_name in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, img_name)

        img_list = glob.glob(img_dir+'*/*.png')


        img_idx = list(range(len(img_list)))
        imgs = {'filename':img_list,'identity':img_idx}
        df_img = pd.DataFrame(imgs)

        df_img_partial = df_img

        def func(fname):
            img = cv2.imread(fname)
            return simple_hist_predictor(img,channel=2,thres=100)
        labels = mmcv.track_parallel_progress(func, df_img_partial['filename'], 8)
        df_img_partial['label'] = labels
        print("==> distribution for hist class")
        print(df_img_partial['label'].value_counts())
    if 1:
        root_dir = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/image_B_v1.1/'
        query_dir = root_dir + 'gallery/'
        #gallery_dir = root_dir + 'gallery/'
        # root_dir = '/data/Dataset/PReID/rep_dataset/'
        # query_dir = root_dir + 'rep_f0_query/'
        # gallery_dir = root_dir + 'rep_f0_gallery/'

        query_list = glob.glob(query_dir+'*.png')
       # gallery_list = glob.glob(gallery_dir+'*.png')
       
        #query_list = []
        gallery_list = []
        query_list += gallery_list

        query_idx = list(range(len(query_list)))
        querys = {'filename':query_list,'identity':query_idx}
        df_query = pd.DataFrame(querys)

        df_query_partial = df_query

        def func(fname):
            img = cv2.imread(fname)
            #return simple_hist_predictor(img,channel=2,thres=100)
            return compute(img)
        labels = mmcv.track_parallel_progress(func, df_query_partial['filename'], 8)
        df_query_partial['label'] = labels
        print("==> distribution for hist class")
        print(df_query_partial['label'].value_counts())

        for i, each in enumerate(range(len(df_query_partial))):
            filename = df_query_partial['filename'][each]
            label =  df_query_partial['label'][each]
            if label == 0 :
                shutil.copy(filename, os.path.join('/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/image_B_v1.1/gallery_green', filename.split('/')[-1]))
            else:
                shutil.copy(filename, os.path.join('/data/xuzihao/NAIC/ReID/data/NAIC_2020/rematch/image_B_v1.1/gallery_normal', filename.split('/')[-1]))
