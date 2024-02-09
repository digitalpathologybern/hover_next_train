import os
import numpy as np
import json
import math 
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
import pandas as pd
from tqdm.auto import tqdm

def euclidean_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def main(out_path, root, val_root):
    out_stats = []
    val_root = os.path.join(val_root,"labels")
    cls = 5 # eosinophils
    for roi in tqdm(sorted(os.listdir(val_root))):
        id_ = os.path.splitext(roi)[0].split("_")[1]
        gt_cent = np.loadtxt(os.path.join(val_root, roi),delimiter=",")
        with open(os.path.join(root,f"{id_}_img/class_inst.json"),"r") as f:
            pred_cent = np.array([[v[0],*v[1][1:]] for v in json.load(f).values()])
        
        gt_tree = KDTree(gt_cent)
        pred_tree = KDTree(pred_cent[:,1:])

        dst_ = gt_tree.query_ball_tree(pred_tree, 6)
        pairwise_cent_dist = np.full(
                    [len(gt_cent), len(pred_cent)], 100, dtype=np.float64
                )
        for i, j in enumerate(dst_):
            for k in j:
                pairwise_cent_dist[i, k] = euclidean_dist(gt_cent[i], pred_cent[k,1:])
        match_euc_dist = 6
        paired_true, paired_pred = linear_sum_assignment(pairwise_cent_dist)
        paired_cen = pairwise_cent_dist[paired_true, paired_pred]
        paired_true = list(paired_true[paired_cen < match_euc_dist] + 1)
        paired_pred = list(paired_pred[paired_cen < match_euc_dist] + 1)
        paired_cen = paired_cen[paired_cen < match_euc_dist]
        unpaired_true = [idx for idx in np.arange(len(gt_cent)) if idx not in paired_true]
        unpaired_pred = [idx for idx in np.arange(len(pred_cent)) if idx not in paired_pred]
        class_pairs = []
        for i, j in zip(paired_true, paired_pred):
            class_pairs.append((5, pred_cent[j-1,0]))
        class_pairs = np.array(class_pairs)


        add_fp = []
        for i in unpaired_pred:
            add_fp.append(pred_cent[i-1,0])
        add_fp = np.array(add_fp)
        add_fn = np.full(len(unpaired_true), 5)
        

        t = class_pairs[:, 0] == cls
        p = class_pairs[:, 1] == cls
        t_n = class_pairs[:, 0] != cls
        p_n = class_pairs[:, 1] != cls

        tp_c = np.count_nonzero(t & p)
        fp_c = np.count_nonzero(t_n & p) + np.count_nonzero(add_fp == cls)
        fn_c = np.count_nonzero(t & p_n) + np.count_nonzero(add_fn == cls)
        tn_c = np.count_nonzero(t_n & p_n)
        out_stats.append({
                    "id": int(id_),
                    "class": "eosinophil",
                    "TP": tp_c,
                    "FP": fp_c,
                    "FN": fn_c,
                    "TN": tn_c,
                })
    group_these = [[0,1],[2],[3],[4,5,6],[7],[8],[9,10]]
    df = pd.DataFrame(out_stats)
    for i,g in enumerate(group_these):
        df.loc[df["id"].isin(g),"group"] = i
    grp_df = df.groupby("group").sum().reset_index()
    grp_df["f1"] = 2*grp_df["TP"]/(2*grp_df["TP"]+grp_df["FP"]+grp_df["FN"])
    grp_df["precision"] = grp_df["TP"]/(grp_df["TP"]+grp_df["FP"])
    grp_df["recall"] = grp_df["TP"]/(grp_df["TP"]+grp_df["FN"])
    grp_df[["group","TP","FP","FN","f1","precision","recall"]].to_csv(out_path, index=False)   

if __name__ == "__main__":
    '''
    To use this, you must have ran inference on all ROI crops from the eos_val dataset and stored the results.
    This requires the HoVer-NeXt inference repository!
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="eos_val.csv")
    parser.add_argument("--root", type=str, required=True, help="path to output directory for all rois")
    parser.add_argument("--val_root", type=str, required=True, help="path to eos_val dataset (e.h. '/path-to-/eos_val/')")
    args = parser.parse_args()
    main(args.out, args.root, args.val_root)
