# -*- coding: utf-8 -*-
import os, shutil
import numpy as np
from easydict import EasyDict as edict
import argparse
import os.path as osp
import ipdb


parser = argparse.ArgumentParser(description='clustering')
parser.add_argument('--root', type=str, default='data', help="root path to data directory")
parser.add_argument('--subset', type=str, default='0')
parser.add_argument('--thres', type=float, default=0.5, help="threshold for split")
args = parser.parse_args()


raw_path = osp.join(args.root, args.subset)
clusters_path = osp.join(args.root, args.subset+'_clusters')

counter = 0
clusters = []
walktree = os.walk(raw_path)
for dirname, subdirs, files in walktree:
    for fi in files:
        if(fi.endswith('.vec')):
            filename = osp.join(dirname, fi)
            vec = np.loadtxt(filename)
            if vec.shape[0] == 0: continue
            vec = np.array(vec[1:])
            vec_norm = np.sqrt(np.dot(vec, vec))
            vec = vec / vec_norm
            similarity = None
            for cluster in clusters:
                simi = np.matmul(cluster.mf, vec.T)
                mean_simi = simi.mean()
                if similarity is None:
                    similarity = mean_simi
                else:
                    similarity = np.vstack((similarity, mean_simi))
            if similarity is None:
                max_simi = 0
            else:
                max_simi = np.amax(similarity)
                idx = np.argmax(similarity)

            print (counter, filename, max_simi)
            counter += 1

            if(max_simi > args.thres):
                clusters[idx].fea = np.vstack((clusters[idx].fea, vec))
                clusters[idx].mf = clusters[idx].fea.mean(0)
                clusters[idx].dirname.append(dirname)
                clusters[idx].filename.append(fi)
            else:
                cluster = edict()
                cluster.fea = vec
                cluster.mf = vec
                cluster.dirname = []
                cluster.filename = []
                cluster.dirname.append(dirname)
                cluster.filename.append(fi)
                clusters.append(cluster)
                print('Now cluster size is %d'%(len(clusters)))


# change the directory name as the image prefix, every time we transfer the image
# we infer whether the image's prefix has the same directory, if there is, just transfer the img
# to the directory 
id_alloc = 1
for cluster in clusters:
    folder = osp.join(clusters_path, str(id_alloc)) 
    id_alloc += 1
    if(not osp.exists(folder)):
        os.makedirs(folder)
    for i in range(len(cluster.filename)):
        img_dir = cluster.dirname[i]
        vec_file = cluster.filename[i]
        img_file = vec_file.replace('.vec', '.png')
        dst_vec_file = osp.join(folder, vec_file)
        dst_img_file = osp.join(folder, img_file)
        if not osp.exists(osp.join(img_dir, vec_file)): continue
        if not osp.exists(osp.join(img_dir, img_file)): continue
        shutil.copyfile(osp.join(img_dir, vec_file), dst_vec_file)	
        shutil.copyfile(osp.join(img_dir, img_file), dst_img_file)	

