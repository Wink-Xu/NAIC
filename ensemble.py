import json
from collections import defaultdict
import glob
import torch
import numpy as np
import os


def weightSum(paths, weights=None):
    print('ensemble ...')
    results = [torch.load(path, map_location='cpu') for path in paths]

    anchor = results[0]
    anchor_query_path = anchor['query_path']
    anchor_query_path = list(map(os.path.basename, anchor_query_path))
    anchor_gallery_path = anchor['gallery_path']
    anchor_gallery_path = list(map(os.path.basename, anchor_gallery_path))
    anchor_dist_mat = anchor['dist_mat']

    anchor_query_path = np.asarray(anchor_query_path, np.str)
    anchor_gallery_path = np.asarray(anchor_gallery_path, np.str)

    query_idx = np.argsort(anchor_query_path)
    anchor_query_path = anchor_query_path[query_idx]

    gallery_idx = np.argsort(anchor_gallery_path)
    anchor_gallery_path = anchor_gallery_path[gallery_idx]

    anchor_dist_mat = anchor_dist_mat[query_idx, :]
    anchor_dist_mat = anchor_dist_mat[:, gallery_idx]

    dist_mats = []
    for result in results:
        query_path = result['query_path']
        gallery_path = result['gallery_path']
        dist_mat = result['dist_mat']
        query_path = np.asarray(query_path, np.str)
        gallery_path = np.asarray(gallery_path, np.str)
        query_idx = np.argsort(query_path)
        assert (query_path[query_idx] != anchor_query_path).sum() == 0
        gallery_idx = np.argsort(gallery_path)
        assert (gallery_path[gallery_idx] != anchor_gallery_path).sum() == 0
        dist_mat = dist_mat[query_idx, :]
        dist_mat = dist_mat[:, gallery_idx]
        dist_mats.append(dist_mat)

    res = np.zeros(anchor_dist_mat.shape)
    if weights is not None:
        for dist_mat, weight in zip(dist_mats, weights):
            res += weight * dist_mat
    else:
        for dist_mat in dist_mats:
            res += dist_mat
        res /= len(dist_mats)
        print('...')
    save_dict = {'dist_mat': res,
                 'query_path': anchor_query_path.tolist(),
                 'gallery_path': anchor_gallery_path.tolist()}

   # torch.save(save_dict, './final_distmat_B.pth')
    _, indices = torch.topk(torch.from_numpy(res), k=200, dim=1, largest=False)
    sumbit = defaultdict(list)
    anchor_query_path = np.asarray(anchor_query_path, np.str)
    anchor_gallery_path = np.asarray(anchor_gallery_path, np.str)
    for i, qpath in enumerate(anchor_query_path):
        img_name = os.path.basename(qpath)
        sumbit[img_name] = anchor_gallery_path[indices[i].numpy()].tolist()
    with open(os.path.join('./result_for_ensemble_C', 'temp.json'), 'w', encoding='utf8') as f:
        json.dump(sumbit, f)


if __name__ == '__main__':
    # model_results = [
    #     'result_for_ensemble_B/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth.pth',
    #     'result_for_ensemble_B/resnet101_ibn_b_32x8_90_s1_576x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth.pth',
    #     'result_for_ensemble_B/resnet101_ibn_b_64x6_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug.pth',
    #     'result_for_ensemble_B/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug.pth',
    #     'result_for_ensemble_B/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_std.pth',
    #     'result_for_ensemble_B/resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth_allData.pth',
    # ]

    model_results = [
        'result_for_ensemble_C/allData1026_resnet101_ibn_b_32x8_90_s1_384x192_rematch_base_notRed.pth',
        'result_for_ensemble_C/allData1026_resnet101_ibn_b_32x8_90_s1_384x192_rematch_base_unlabel.pth',
       # 'result_for_ensemble_C/allData1026_resnet101_ibn_b_32x8_90_s1_384x192_ad2_triplet_gpu1_apex_arcface_gemlr10_2019DataNew_wloss_syncBN_dataAug_noSmooth.pth',
       # 'result_for_ensemble_C/dmt_normal.pth'
    ]

    weights = [0.5, 0.5]


    weightSum(model_results, weights)