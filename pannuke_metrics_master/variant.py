# This file is a modified version of the code in https://github.com/TissueImageAnalytics/PanNuke-metrics
# with no changes to the actual evaluation, rather basic performance improvements.

import numpy as np
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from pannuke_metrics_master.utils import get_fast_pq, remap_label
import os
from tqdm.auto import tqdm

# This is a variant of the same evaluations as in pannuke_metrics_master/run.py
# using ProcessPoolExecutor for faster evaluation during training


def get_metrics(i, pred_list, gt):
    pq = []
    pred_bin = remap_label(pred_list[..., 0].astype(np.int32))
    true_bin = remap_label(gt[..., 0].astype(np.int32))

    if len(np.unique(true_bin)) == 1:
        pq_bin = (
            np.nan
        )  # if ground truth is empty for that class, skip from calculation
    else:
        [_, _, pq_bin], _ = get_fast_pq(true_bin, pred_bin)  # compute PQ

    # loop over the classes
    for j in range(5):
        pred_tmp = pred_list[..., 0].astype(np.int32).copy()
        pred_tmp[pred_list[..., 1] != (j + 1)] = 0
        true_tmp = gt[..., 0].astype(np.int32).copy()
        true_tmp[gt[..., 1] != (j + 1)] = 0
        pred_tmp = remap_label(pred_tmp)
        true_tmp = remap_label(true_tmp)

        if len(np.unique(true_tmp)) == 1:
            pq_tmp = (
                np.nan
            )  # if ground truth is empty for that class, skip from calculation
        else:
            [_, _, pq_tmp], _ = get_fast_pq(true_tmp, pred_tmp)  # compute PQ

        pq.append(pq_tmp)
    return i, pq, [pq_bin]


def get_pannuke_pq(gt, pred, types=None):
    mPQ_all = []
    bPQ_all = []
    print("running pannuke eval...")
    with ProcessPoolExecutor(max_workers=4) as executor:
        future_to_metrics = []
        for i in range(len(gt)):
            future_to_metrics.append(executor.submit(get_metrics, i, pred[i], gt[i]))

        for future in concurrent.futures.as_completed(future_to_metrics):
            i_, mpq, pq = future.result()
            mPQ_all.append((i_, mpq))
            bPQ_all.append((i_, pq))

    mPQ_all = [x[1] for x in sorted(mPQ_all, key=lambda x: x[0])]
    bPQ_all = [x[1] for x in sorted(bPQ_all, key=lambda x: x[0])]

    # using np.nanmean skips values with nan from the mean calculation
    mPQ_each_image = [np.nanmean(pq) for pq in mPQ_all]
    bPQ_each_image = [np.nanmean(pq_bin) for pq_bin in bPQ_all]

    # class metric
    neo_PQ = np.nanmean([pq[0] for pq in mPQ_all])
    inflam_PQ = np.nanmean([pq[1] for pq in mPQ_all])
    conn_PQ = np.nanmean([pq[2] for pq in mPQ_all])
    dead_PQ = np.nanmean([pq[3] for pq in mPQ_all])
    nonneo_PQ = np.nanmean([pq[4] for pq in mPQ_all])

    tissue_types = [
        "Adrenal_gland",
        "Bile-duct",
        "Bladder",
        "Breast",
        "Cervix",
        "Colon",
        "Esophagus",
        "HeadNeck",
        "Kidney",
        "Liver",
        "Lung",
        "Ovarian",
        "Pancreatic",
        "Prostate",
        "Skin",
        "Stomach",
        "Testis",
        "Thyroid",
        "Uterus",
    ]

    # Print for each class
    print("Printing calculated metrics on a single split")
    print("-" * 40)
    print("Neoplastic PQ: {}".format(neo_PQ))
    print("Inflammatory PQ: {}".format(inflam_PQ))
    print("Connective PQ: {}".format(conn_PQ))
    print("Dead PQ: {}".format(dead_PQ))
    print("Non-Neoplastic PQ: {}".format(nonneo_PQ))
    print("-" * 40)
    all_tissue_mPQ = None
    all_tissue_bPQ = None
    if types is not None:
        # Print for each tissue
        all_tissue_mPQ = {}
        all_tissue_bPQ = {}
        for tissue_name in tissue_types:
            indices = [i for i, x in enumerate(types) if x == tissue_name]
            tissue_PQ = [mPQ_each_image[i] for i in indices]
            # print("{} PQ: {} ".format(tissue_name, np.nanmean(tissue_PQ)))
            tissue_PQ_bin = [bPQ_each_image[i] for i in indices]
            # print("{} PQ binary: {} ".format(tissue_name, np.nanmean(tissue_PQ_bin)))
            all_tissue_mPQ[tissue_name] = np.nanmean(tissue_PQ)
            all_tissue_bPQ[tissue_name] = np.nanmean(tissue_PQ_bin)
        # Show overall metrics - mPQ is average PQ over the classes and the tissues, bPQ is average binary PQ over the tissues
        at_mpq = np.nanmean(list(all_tissue_mPQ.values()))
        at_bpq = np.nanmean(list(all_tissue_bPQ.values()))
        print("-" * 40)
        print("Average mPQ:{}".format(at_mpq))
        print("Average bPQ:{}".format(at_bpq))

    return (
        np.nanmean([neo_PQ, inflam_PQ, conn_PQ, dead_PQ, nonneo_PQ]),
        np.nanmean(bPQ_each_image),
        [neo_PQ, inflam_PQ, conn_PQ, dead_PQ, nonneo_PQ],
        [all_tissue_mPQ, all_tissue_bPQ],
    )
