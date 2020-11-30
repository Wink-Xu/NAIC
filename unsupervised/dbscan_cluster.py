import cv2
import os
import numpy as np
import json
import time 
import pprint
import shutil

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader,SequentialSampler,RandomSampler
from sklearn.cluster import DBSCAN
from scipy import sparse


# def euclidean_distance(x, y):
#     """
#     Args:
#       x: pytorch Variable, with shape [m, d]
#       y: pytorch Variable, with shape [n, d]
#     Returns:
#       dist: pytorch Variable, with shape [m, n]
#     """
#     m, n = x.size(0), y.size(0)
#     xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
#     yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
#     dist = xx + yy
#     dist.addmm_(1, -2, x, y.t())
#     dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
#     return dist

def euclidean_distance(qf, gf):

    m = qf.shape[0]
    n = gf.shape[0]

    # dist_mat = torch.pow(qf,2).sum(dim=1, keepdim=True).expand(m,n) +\
    #     torch.pow(gf,2).sum(dim=1, keepdim=True).expand(n,m).t()
    # dist_mat.addmm_(1,-2,qf,gf.t())

    # for L2-norm feature
    dist_mat = 2 - 2 * torch.matmul(qf, gf.t())
    return dist_mat.clamp(min=1e-12).sqrt()


def predict_pseudo_label(sparse_distmat, eps=0.5, min_points=4, max_points=50,algorithm='brute'):
    dbscaner = DBSCAN(eps = eps, min_samples = min_points,algorithm=algorithm,n_jobs=6,metric='precomputed')
    # dbscaner = DBSCAN(eps = eps, min_samples = min_points,n_jobs=6,metric='precomputed')
    cls_res = dbscaner.fit_predict(sparse_distmat)
    res_dict = dict()
    for i in range(cls_res.shape[0]):
        if cls_res[i] == -1 or cls_res[i] == None:
            continue
        if cls_res[i] not in res_dict.keys():
            res_dict[cls_res[i]] = []

        res_dict[cls_res[i]].append(i)
    filter_res = {}
    for k , v in res_dict.items():
        if len(v) >= min_points and len(v) <= max_points:
            filter_res[k] = v
    # import pdb;pdb.set_trace()
    
    return filter_res

       
def get_sparse_distmat(all_feature,eps,len_slice = 1000,use_gpu=False,dist_k=-1,top_k=35):

    
    if use_gpu:
        gpu_feature = all_feature.cuda()
    else:
        gpu_feature = all_feature
    n_iter = len(all_feature) // len_slice + int(len(all_feature) % len_slice > 0)
    distmats = []
    kdist = []
    with tqdm(total=n_iter) as pbar:
        for i in range(n_iter):
            if use_gpu:
                distmat = euclidean_distance(gpu_feature[i*len_slice:(i+1)*len_slice], gpu_feature).data.cpu().numpy()
            else:
                distmat = euclidean_distance(gpu_feature[i*len_slice:(i+1)*len_slice], gpu_feature).numpy()

            if dist_k>0:
                dist_rank = np.argpartition(distmat,range(1,dist_k+1)) # 1,N
                for j in range(distmat.shape[0]):
                    kdist.append(distmat[j,dist_rank[j,dist_k]])
            if 0:
                initial_rank = np.argpartition(distmat,top_k) # 1,N
                for j in range(distmat.shape[0]):
                    distmat[j,initial_rank[j,top_k:]] = 0
            else:
                distmat[distmat>eps] = 0
            distmats.append(sparse.csr_matrix(distmat))
     #      import pdb;pdb.set_trace()
            pbar.update(1)
    if dist_k>0:
        return sparse.vstack(distmats),kdist

    return sparse.vstack(distmats)

def inference_cluster():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

    pseudo_eps = 0.6
    pseudo_minpoints = 2
    pseudo_maxpoints = 1000
    pseudo_algorithm = 'brute'
    print("begin")
    vec_path = '../../data/NAIC_2020/rematch/unlabel/images'
    #vec_path = '/data/xuzihao/NAIC/ReID/data/NAIC_2020/train/noLabelImg'
    vecName = []
    feats = []
    cnt = 0
    for i in tqdm(os.listdir(vec_path)):
        if i.endswith('.vec'):
            vecName.append(i)
            file_path = os.path.join(vec_path, i)
            vec = np.loadtxt(file_path)[1:] 
            vec_norm = np.sqrt(np.dot(vec, vec))
            vec = vec / vec_norm
            feat = torch.FloatTensor(vec).unsqueeze(0)
            feats.append(feat)
            cnt += 1

    all_feature = torch.cat(feats, dim=0)

    print("==> using pseudo eps:{} minPoints:{} maxpoints:{}".format(pseudo_eps, pseudo_minpoints, pseudo_maxpoints))

    st = time.time()
    
    all_feature = F.normalize(all_feature, p=2, dim=1)


    all_distmat,kdist = get_sparse_distmat(all_feature,eps=pseudo_eps+0.05,len_slice=1000,use_gpu=True,dist_k=pseudo_minpoints)
    # plt.plot(list(range(len(kdist))),np.sort(kdist),linewidth=0.5)
    # plt.savefig('test_kdist.png')          

    pseudolabels = predict_pseudo_label(all_distmat,pseudo_eps,pseudo_minpoints,pseudo_maxpoints,pseudo_algorithm)
    print("pseudo cost: {}s".format(time.time()-st))
    print("pseudo id cnt:",len(pseudolabels))
    print("pseudo img cnt:",len([x for k,v in pseudolabels.items() for x in v]))

    # # save
    all_list = vecName
    save_path = '../../data/NAIC_2020/rematch/unlabel/noLabelImg_dbscan_0.6_2_1000'

    pid = 0
    camid = 0
    nignore_query = 0
    for k,v in pseudolabels.items():
        os.makedirs(os.path.join(save_path, str(pid)),exist_ok=True)
        # [fileter]
        # query_cnt = 0
        # for _index in pseudolabels[k]:
        #     if _index<len(query_list):
        #         query_cnt += 1
        # if query_cnt>=4:
        #     nignore_query += 1
        #     continue
        for _index in pseudolabels[k]:
            filename = all_list[_index]
            new_filename = all_list[_index].replace('.vec', '.png')
            shutil.copy(os.path.join(vec_path, all_list[_index].replace('.vec', '.png')), os.path.join(save_path, str(pid), new_filename))
            camid += 1
        pid += 1
    print("pseudo ignore id cnt:",nignore_query)

if __name__ == "__main__":
    inference_cluster()