import numpy as np
import pathlib
import os
import shutil
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import r2_score
from scipy.ndimage import find_objects
import scipy.ndimage as ndi
import os
import numpy as np
from skimage.measure import regionprops, label
import cv2
from scipy.spatial.distance import directed_hausdorff
import math
import cv2
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

# code copied from
# https://github.com/TissueImageAnalytics/CoNIC/blob/main/metrics/stats_utils.py
# https://github.com/TissueImageAnalytics/CoNIC/blob/main/misc/utils.py


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3]
    not [0, 2, 4, 6]. The ordering of instances (which one comes first)
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.
    Args:
        pred (ndarray): the 2d array contain instances where each instances is marked
            by non-zero integer.
        by_size (bool): renaming such that larger nuclei have a smaller id (on-top).
    Returns:
        new_pred (ndarray): Array with continguous ordering of instances.
    """
    pred_id = list(np.unique(pred))
    try:
        pred_id.remove(0)
    except ValueError:
        pass
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def cropping_center(x, crop_shape, batch=False):
    """Crop an array at the centre with specified dimensions."""
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def rm_n_mkdir(dir_path):
    """Remove and make directory."""
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def recur_find_ext(root_dir, ext_list):
    """Recursively find all files in directories end with the `ext` such as `ext='.png'`.
    Args:
        root_dir (str): Root directory to grab filepaths from.
        ext_list (list): File extensions to consider.
    Returns:
        file_path_list (list): sorted list of filepaths.
    """
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def get_bounding_box(img):
    """Get the bounding box coordinates of a binary input- assumes a single object.
    Args:
        img: input binary image.
    Returns:
        bounding box coordinates
    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def get_multi_pq_info(true, pred, nr_classes=6, match_iou=0.5):
    """Get the statistical information needed to compute multi-class PQ.

    CoNIC multiclass PQ is achieved by considering nuclei over all images at the same time,
    rather than averaging image-level results, like was done in MoNuSAC. This overcomes issues
    when a nuclear category is not present in a particular image.

    Args:
        true (ndarray): HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map.
        pred: HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map.
        nr_classes (int): Number of classes considered in the dataset.
        match_iou (float): IoU threshold for determining whether there is a detection.

    Returns:
        statistical info per class needed to compute PQ.

    """

    assert match_iou >= 0.0, "Cant' be negative"

    true_inst = true[..., 0]
    pred_inst = pred[..., 0]
    ###
    true_class = true[..., 1]
    pred_class = pred[..., 1]

    pq = []
    for idx in range(nr_classes):
        pred_class_tmp = pred_class == idx + 1
        pred_inst_oneclass = pred_inst * pred_class_tmp
        pred_inst_oneclass = remap_label(pred_inst_oneclass)
        ##
        true_class_tmp = true_class == idx + 1
        true_inst_oneclass = true_inst * true_class_tmp
        true_inst_oneclass = remap_label(true_inst_oneclass)

        pq_oneclass_info = get_pq(true_inst_oneclass, pred_inst_oneclass, remap=False)

        # add (in this order) tp, fp, fn iou_sum
        pq_oneclass_stats = [
            pq_oneclass_info[1][0],
            pq_oneclass_info[1][1],
            pq_oneclass_info[1][2],
            pq_oneclass_info[2],
        ]
        pq.append(pq_oneclass_stats)

    return pq


def get_pq(true, pred, match_iou=0.5, remap=True):
    """Get the panoptic quality result.

    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4]
    not [2, 3, 6, 10]. Please call `label` beforehand. Here, the `by_size` flag
    has no effect on the result.
    Args:
        true (ndarray): HxW ground truth instance segmentation map
        pred (ndarray): HxW predicted instance segmentation map
        match_iou (float): IoU threshold level to determine the pairing between
            GT instances `p` and prediction instances `g`. `p` and `g` is a pair
            if IoU > `match_iou`. However, pair of `p` and `g` must be unique
            (1 prediction instance to 1 GT instance mapping). If `match_iou` < 0.5,
            Munkres assignment (solving minimum weight matching in bipartite graphs)
            is caculated to find the maximal amount of unique pairing. If
            `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
            the number of pairs is also maximal.
        remap (bool): whether to ensure contiguous ordering of instances.

    Returns:
        [dq, sq, pq]: measurement statistic
        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

        paired_iou.sum(): sum of IoU within true positive predictions

    """
    assert match_iou >= 0.0, "Cant' be negative"
    # ensure instance maps are contiguous
    if remap:
        pred, _ = ndi.label(pred)
        true, _ = ndi.label(true)

    true = np.copy(true)
    pred = np.copy(pred)
    true = true.astype("int32")
    pred = pred.astype("int32")
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    # caching pairwise iou
    true_slices = find_objects(true)
    pred_slices = find_objects(pred)
    for true_id, slc in enumerate(true_slices):
        t_mask_lab = true[slc] == (true_id + 1)
        y, x = slc
        pred_true_overlap = pred[slc].copy()
        valid = np.unique(pred_true_overlap[t_mask_lab])
        for pred_id in valid:
            if pred_id == 0:
                continue
            slc_ = pred_slices[pred_id - 1]
            if slc_ is None:
                continue
            y_, x_ = slc_
            fin_slc = (
                slice(min(y.start, y_.start), max(y.stop, y_.stop), None),
                slice(min(x.start, x_.start), max(x.stop, x_.stop), None),
            )
            t_mask_crop2 = (true[fin_slc] == (true_id + 1)).astype(int)
            p_mask_crop2 = (pred[fin_slc] == (pred_id)).astype(int)

            total = (t_mask_crop2 + p_mask_crop2).sum()
            inter = (t_mask_crop2 * p_mask_crop2).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id, pred_id - 1] = iou
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    try:
        dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    except ZeroDivisionError:
        dq = np.NaN
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return (
        [dq, sq, dq * sq],
        [tp, fp, fn],
        paired_iou.sum(),
    )


def get_multi_r2(true, pred, class_names):
    """Get the correlation of determination for each class and then
    average the results.

    Args:
        true (pd.DataFrame): dataframe indicating the nuclei counts for each image and category.
        pred (pd.DataFrame): dataframe indicating the nuclei counts for each image and category.

    Returns:
        multi class coefficient of determination

    """
    # first check to make sure that the appropriate column headers are there
    for col in true.keys():
        if col not in class_names:
            raise ValueError("%s column header not recognised")

    for col in pred.keys():
        if col not in class_names:
            raise ValueError("%s column header not recognised")

    # for each class, calculate r2 and then take the average
    r2_list = []
    for class_ in class_names:
        true_oneclass = true[class_].tolist()
        pred_oneclass = pred[class_].tolist()
        r2_list.append(r2_score(true_oneclass, pred_oneclass))

    print(r2_list)
    return np.mean(np.array(r2_list)), np.array(r2_list)


def calc_MPQ(pred_list, gt_list, nclasses=6):
    mode = "seg_class"
    pred_array = np.stack(pred_list, axis=0)
    true_array = np.stack(gt_list, axis=0)

    seg_metrics_names = ["pq", "multi_pq+"]
    reg_metrics_names = ["r2"]

    all_metrics = {}
    if mode == "seg_class":
        pq_list = []
        mpq_info_list = []

        nr_patches = pred_array.shape[0]

        for patch_idx in range(nr_patches):
            # get a single patch
            pred = pred_array[patch_idx]
            true = true_array[patch_idx]

            # instance segmentation map
            pred_inst = pred[..., 0]
            true_inst = true[..., 0]

            # ===============================================================

            for idx, metric in enumerate(seg_metrics_names):
                if metric == "pq":
                    # get binary panoptic quality
                    pq = get_pq(true_inst, pred_inst)
                    pq = pq[0][2]
                    pq_list.append(pq)
                elif metric == "multi_pq+":
                    # get the multiclass pq stats info from single image
                    mpq_info_single = get_multi_pq_info(true, pred, nclasses)
                    mpq_info = []
                    # aggregate the stat info per class
                    for single_class_pq in mpq_info_single:
                        tp = single_class_pq[0]
                        fp = single_class_pq[1]
                        fn = single_class_pq[2]
                        sum_iou = single_class_pq[3]
                        mpq_info.append([tp, fp, fn, sum_iou])
                    mpq_info_list.append(mpq_info)
                else:
                    raise ValueError("%s is not supported!" % metric)

        pq_metrics = np.array(pq_list)
        pq_metrics_avg = np.nanmean(pq_metrics, axis=-1)  # average over all images
        if "multi_pq+" in seg_metrics_names:
            print("debug", mpq_info_list[0])
            mpq_info_metrics = np.array(mpq_info_list, dtype="float")
            # sum over all the images
            total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)

        for idx, metric in enumerate(seg_metrics_names):
            if metric == "multi_pq+":
                mpq_list = []
                # for each class, get the multiclass PQ
                for cat_idx in range(total_mpq_info_metrics.shape[0]):
                    total_tp = total_mpq_info_metrics[cat_idx][0]
                    total_fp = total_mpq_info_metrics[cat_idx][1]
                    total_fn = total_mpq_info_metrics[cat_idx][2]
                    total_sum_iou = total_mpq_info_metrics[cat_idx][3]

                    # get the F1-score i.e DQ
                    dq = total_tp / (
                        (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6
                    )
                    # get the SQ, when not paired, it has 0 IoU so does not impact
                    sq = total_sum_iou / (total_tp + 1.0e-6)
                    mpq_list.append(dq * sq)
                mpq_metrics = np.array(mpq_list)
                all_metrics[metric] = [np.mean(mpq_metrics)]
            else:
                all_metrics[metric] = [pq_metrics_avg]

    df = pd.DataFrame(all_metrics)
    print(df)
    print(mpq_list)
    return df, mpq_list


# METRICS REWORK:


def crop(t, croph, cropw):
    h, w = t.shape[-3:-1]
    startw = w // 2 - (cropw // 2)
    starth = h // 2 - (croph // 2)
    return t[:, starth : starth + croph, startw : startw + cropw]


def f1_custom(gt, pred):
    tp = np.count_nonzero(gt * pred)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(gt & ~pred)
    try:
        f1 = (2 * tp) / ((2 * tp) + fp + fn + 1e-6)
    except ZeroDivisionError:
        f1 = np.NaN
    return f1


def mcc_custom(gt, pred):
    tp = np.count_nonzero(gt * pred)
    fp = np.count_nonzero(pred & ~gt)
    fn = np.count_nonzero(gt & ~pred)
    tn = np.count_nonzero(~gt & ~pred)
    try:
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
    except ZeroDivisionError:
        mcc = np.NaN
    return mcc


def euclidean_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def expand_bbox(bb, size_px=3, shape_constraints=None):
    """Expand bounding box by size_px in all directions.

    Args:
        bb (tuple): Bounding box coordinates (min_row, min_col, max_row, max_col).
        size_px (int): Number of pixels to expand in all directions.
        shape_constraints (tuple): (max_row, max_col) to limit the size of the bounding box.

    Returns:
        tuple: Expanded bounding box coordinates (min_row, min_col, max_row, max_col).
    """
    min_row, min_col, max_row, max_col = bb
    min_row = max(min_row - size_px, 0)
    min_col = max(min_col - size_px, 0)
    max_row = (
        min(max_row + size_px, shape_constraints[0] - 1)
        if shape_constraints
        else max_row + size_px
    )
    max_col = (
        min(max_col + size_px, shape_constraints[1] - 1)
        if shape_constraints
        else max_col + size_px
    )
    return min_row, min_col, max_row, max_col


def merge_bboxes(bbi, bbj):
    return (
        min(bbi[0], bbj[0]),
        min(bbi[1], bbj[1]),
        max(bbi[2], bbj[2]),
        max(bbi[3], bbj[3]),
    )


def get_class(regprop, cls_map):
    return cls_map[
        regprop.bbox[0] : regprop.bbox[2], regprop.bbox[1] : regprop.bbox[3]
    ][regprop.image][0]


def get_contours(mask):
    contours = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )[0]
    if len(contours) > 1:
        return np.squeeze(np.concatenate(contours))
    else:
        contours = np.squeeze(contours)
        if len(contours.shape) == 1:
            return contours[np.newaxis, :]
        else:
            return contours


def per_tile_worker(cnt, gt_tile, pred_tile, match_euc_dist, class_names):
    gt_inst = label(gt_tile[..., 0], connectivity=2)
    pred_inst = label(pred_tile[..., 0], connectivity=2)
    # get simple foreground background segmentation metrics
    gt_bin = (gt_inst > 0).flatten()
    pred_bin = (pred_inst > 0).flatten()
    tile_f1 = f1_custom(gt_bin, pred_bin)
    tile_mcc = mcc_custom(gt_bin, pred_bin)

    true_id_list = np.arange(np.max(gt_inst))
    pred_id_list = np.arange(np.max(pred_inst))
    # pairwise_hd = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)
    pairwise_cent_dist = np.full(
        [len(true_id_list), len(pred_id_list)], 1000, dtype=np.float64
    )

    true_objects = regionprops(gt_inst)
    pred_objects = regionprops(pred_inst)
    for ti, o in enumerate(true_objects):
        bb = expand_bbox(o.bbox, size_px=2, shape_constraints=gt_inst.shape)
        # gt_mask = gt[c,...,0][bb[0]:bb[2], bb[1]:bb[3]]==o.label
        valid = np.unique(pred_inst[bb[0] : bb[2], bb[1] : bb[3]])
        for pred_id in valid:
            if pred_id == 0:
                continue
            pred_obj = pred_objects[pred_id - 1]
            pairwise_cent_dist[ti, pred_id - 1] = euclidean_dist(
                o.centroid, pred_obj.centroid
            )

    paired_true, paired_pred = linear_sum_assignment(pairwise_cent_dist)
    paired_cen = pairwise_cent_dist[paired_true, paired_pred]
    paired_true = list(paired_true[paired_cen < match_euc_dist] + 1)
    paired_pred = list(paired_pred[paired_cen < match_euc_dist] + 1)
    paired_cen = paired_cen[paired_cen < match_euc_dist]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    try:
        f1 = (2 * tp) / ((2 * tp) + fp + fn)
        f1_d = (2 * tp) / ((2 * tp) + (2 * fp) + (2 * fn))
    except ZeroDivisionError:
        # this means neither on GT nor pred there is a nucleus
        f1 = np.NaN
        f1_d = np.NaN
        return [
            {
                "id": cnt,
                "class": "all",
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "F1": f1,
                "F1_d": f1_d,
                "pixel_f1": tile_f1,
                "pixel_mcc": tile_mcc,
            }
        ]

    # balanced_acc = (tp/(tp+fn) + tp/(tp+fp))/2
    paired_hd = []
    paired_dice = []
    paired_mcc = []
    class_pairs = []
    for i, j in zip(paired_true, paired_pred):
        oi = true_objects[i - 1]
        oj = pred_objects[j - 1]
        oi_class = get_class(oi, gt_tile[..., 1])
        oj_class = get_class(oj, pred_tile[..., 1])

        comb_bb = merge_bboxes(oi.bbox, oj.bbox)
        gt_mask = gt_inst[comb_bb[0] : comb_bb[2], comb_bb[1] : comb_bb[3]] == oi.label
        pred_mask = (
            pred_inst[comb_bb[0] : comb_bb[2], comb_bb[1] : comb_bb[3]] == oj.label
        )

        gt_contours = get_contours(gt_mask)
        pred_contours = get_contours(pred_mask)
        general_hausdorff = max(
            [
                directed_hausdorff(gt_contours, pred_contours)[0],
                directed_hausdorff(pred_contours, gt_contours)[0],
            ]
        )
        paired_hd.append(general_hausdorff)
        paired_dice.append(f1_custom(gt_mask.flatten(), pred_mask.flatten()))
        paired_mcc.append(mcc_custom(gt_mask.flatten(), pred_mask.flatten()))
        class_pairs.append((oi_class, oj_class))

    class_pairs = np.array(class_pairs)
    paired_hd = np.array(paired_hd)
    paired_dice = np.array(paired_dice)
    paired_mcc = np.array(paired_mcc)

    add_fp = []
    for i in unpaired_pred:
        o = pred_objects[i - 1]
        o_class = get_class(o, pred_tile[..., 1])
        add_fp.append(o_class)
    add_fp = np.array(add_fp)
    add_fn = []
    for i in unpaired_true:
        o = true_objects[i - 1]
        o_class = get_class(o, gt_tile[..., 1])
        add_fn.append(o_class)
    add_fn = np.array(add_fn)

    sub_metrics = []
    for cls in np.arange(1, len(class_names) + 1):
        try:
            t = class_pairs[:, 0] == cls
            p = class_pairs[:, 1] == cls
            t_n = class_pairs[:, 0] != cls
            p_n = class_pairs[:, 1] != cls

            tp_hd = np.nanmean(paired_hd[t & p])
            tp_dice = np.nanmean(paired_dice[t & p])
            tp_mcc = np.nanmean(paired_mcc[t & p])
            tp_c = np.count_nonzero(t & p)
            fp_c = np.count_nonzero(t_n & p) + np.count_nonzero(add_fp == cls)
            fn_c = np.count_nonzero(t & p_n) + np.count_nonzero(add_fn == cls)
            tn_c = np.count_nonzero(t_n & p_n)
        except IndexError:
            # fix no match for any class
            tp_hd = np.NaN
            tp_dice = np.NaN
            tp_mcc = np.NaN
            tp_c = 0
            fp_c = np.count_nonzero(add_fp == cls)
            fn_c = np.count_nonzero(add_fn == cls)
            tn_c = 0

        try:
            f1_c = (2 * tp_c) / ((2 * tp_c) + fp_c + fn_c)
            f1_c_d = (2 * tp_c) / ((2 * tp_c) + (2 * fp_c) + (2 * fn_c))
        except ZeroDivisionError:
            f1_c = np.NaN
            f1_c_d = np.NaN
        # balanced accurracy needs special treatment
        try:
            tpr_c = tp_c / (tp_c + fn_c)
        except ZeroDivisionError:
            tpr_c = np.NaN
        try:
            tnr_c = tn_c / (tn_c + fp_c)
        except ZeroDivisionError:
            tnr_c = np.NaN
        bal_acc_c = (tpr_c + tnr_c) / 2
        try:
            mcc_c = ((tp_c * tn_c) - (fp_c * fn_c)) / math.sqrt(
                (tp_c + fp_c) * (tp_c + fn_c) * (tn_c + fp_c) * (tn_c + fn_c)
            )
        except ZeroDivisionError:
            mcc_c = np.NaN

        sub_metrics.append(
            {
                "id": cnt,
                "class": class_names[cls - 1],
                "seg_hausdorff_(TP)": tp_hd,
                "seg_dice_(TP)": tp_dice,
                "seg_mcc_(TP)": tp_mcc,
                "TP": tp_c,
                "FP": fp_c,
                "FN": fn_c,
                "TN": tn_c,
                "F1": f1_c,
                "F1_d": f1_c_d,
                "balanced_acc": bal_acc_c,
                "mcc": mcc_c,
            }
        )

    sub_metrics.append(
        {
            "id": cnt,
            "class": "all",
            "seg_hausdorff_(TP)": np.mean(paired_hd),
            "seg_dice_(TP)": np.mean(paired_dice),
            "seg_mcc_(TP)": np.mean(paired_mcc),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "F1": f1,
            "F1_d": f1_d,
            "pixel_f1": tile_f1,
            "pixel_mcc": tile_mcc,
        }
    )
    return sub_metrics


def per_tile_metrics(gt, pred, class_names, match_euc_dist=6):
    metrics = []
    with ProcessPoolExecutor(4) as executor:
        future_metrics = []
        for cnt, (gt_tile, pred_tile) in enumerate(zip(gt, pred)):
            future_metrics.append(
                executor.submit(
                    per_tile_worker,
                    cnt,
                    gt_tile,
                    pred_tile,
                    match_euc_dist,
                    class_names,
                )
            )
        res = [
            future.result()
            for future in concurrent.futures.as_completed(future_metrics)
        ]
    res = [i for i in sorted(res, key=lambda x: x[0]["id"])]
    for i in res:
        metrics.extend(i)
    return metrics


def get_output(metrics, save_path, class_names):
    metrics_df = pd.DataFrame(metrics)
    if save_path is not None:
        out_path, dataset = save_path
        metrics_df.to_csv(
            os.path.join(out_path, f"{dataset}_per_tile_metrics.csv"),
            sep=",",
            index=False,
        )
    abs_counts = (
        metrics_df.groupby("class")
        .sum()
        .reset_index()[["class", "TP", "FP", "FN", "TN"]]
    )
    mean_stat = metrics_df.groupby("class").mean().reset_index()[["class", "F1_d"]]
    abs_counts = abs_counts.merge(mean_stat, on="class")
    tp = abs_counts["TP"].values
    fp = abs_counts["FP"].values
    fn = abs_counts["FN"].values
    tn = abs_counts["TN"].values
    abs_counts["F1"] = (2 * tp) / ((2 * tp) + fp + fn)
    abs_counts["balanced_acc"] = (tp / (tp + fn) + tn / (tn + fp)) / 2
    abs_counts["mcc"] = ((tp * tn) - (fp * fn)) / np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    abs_counts.loc[abs_counts["class"] == "all", ["balanced_acc", "mcc"]] = "NaN"
    per_class_seg = metrics_df[
        ["id", "class", "seg_hausdorff_(TP)", "seg_dice_(TP)", "seg_mcc_(TP)", "TP"]
    ].copy()
    per_class_seg[
        ["seg_hausdorff_(TP)", "seg_dice_(TP)", "seg_mcc_(TP)"]
    ] *= metrics_df[["TP"]].values
    per_class_seg = per_class_seg.groupby("class").sum().reset_index()
    per_class_seg[
        ["seg_hausdorff_(TP)", "seg_dice_(TP)", "seg_mcc_(TP)"]
    ] /= per_class_seg[["TP"]].values
    per_class_seg.drop(["TP", "id"], axis=1, inplace=True)
    pixel_metrics = (
        metrics_df[metrics_df["class"] == "all"][["pixel_f1", "pixel_mcc"]]
        .describe()
        .drop("count")
    )
    pixel_metrics = pixel_metrics.T.reset_index()
    pixel_metrics.columns = ["metric", "mean", "std", "min", ".25", ".50", ".75", "max"]
    corr = metrics_df[["id", "class", "TP", "FP", "FN", "TN"]].copy()
    corr["gt"] = corr["TP"] + corr["FN"]
    corr["pred"] = corr["TP"] + corr["FP"]
    corr_dict = []
    for c in class_names + ["all"]:
        subset = corr[corr["class"] == c].copy()
        spearman = subset[["gt", "pred"]].corr("spearman").loc["gt", "pred"]
        pearson = subset[["gt", "pred"]].corr("pearson").loc["gt", "pred"]
        log_pearson = (
            subset[["gt", "pred"]].apply(np.log).corr("pearson").loc["gt", "pred"]
        )
        corr_dict.append(
            {
                "class": c,
                "spearman": spearman,
                "pearson": pearson,
                "log_pearson": log_pearson,
            }
        )
    metrics_dict = {
        "binary_pixel_metrics": pixel_metrics.to_dict("records"),
        "count_metrics": abs_counts.to_dict("records"),
        "class_wise_seg_metrics": per_class_seg.to_dict("records"),
        "correlations": corr_dict,
    }
    if save_path is not None:
        with open(os.path.join(out_path, f"{dataset}_metrics.json"), "w") as f:
            json.dump(metrics_dict, f)
    print(abs_counts)
    print(per_class_seg)
    return metrics_df, metrics_dict
