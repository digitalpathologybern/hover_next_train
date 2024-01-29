import os
import json
import cv2
import numpy as np
import pandas as pd

# import mahotas as mh
from scipy.ndimage import find_objects
from src.metrics import crop, per_tile_metrics, get_output, calc_MPQ, get_multi_r2
from src.constants import (
    CLASS_NAMES,
    BEST_MIN_THRESHS,
    BEST_MAX_THRESHS,
    MIN_THRESHS_PANNUKE,
    MAX_THRESHS_PANNUKE,
)


from scipy.special import softmax
from skimage.segmentation import watershed
from pannuke_metrics_master.variant import get_pannuke_pq


def get_pp_params(experiments, cp_root, mit_eval=False, eval_metric="mpq"):
    fg_threshs = []
    seed_threshs = []
    for exp in experiments:
        mod_path = os.path.join(cp_root, exp)
        if exp.startswith("pannuke"):
            with open(
                os.path.join(mod_path, "pannuke_test_param_dict.json"), "r"
            ) as js:
                dt = json.load(js)
                fg_threshs.append(dt[f"best_fg_{eval_metric}"])
                seed_threshs.append(dt[f"best_seed_{eval_metric}"])
        elif mit_eval:
            with open(os.path.join(mod_path, "liz_test_param_dict.json"), "r") as js:
                dt = json.load(js)
                fg_tmp = dt[f"best_fg_{eval_metric}"]
                seed_tmp = dt[f"best_seed_{eval_metric}"]
            with open(os.path.join(mod_path, "mit_test_param_dict.json"), "r") as js:
                dt = json.load(js)
                fg_tmp[-1] = dt[f"best_fg_{eval_metric}"][-1]
                seed_tmp[-1] = dt[f"best_seed_{eval_metric}"][-1]
            fg_threshs.append(fg_tmp)
            seed_threshs.append(seed_tmp)
        else:
            with open(os.path.join(mod_path, "param_dict.json"), "r") as js:
                dt = json.load(js)
                fg_threshs.append(dt[f"best_fg_{eval_metric}"])
                seed_threshs.append(dt[f"best_seed_{eval_metric}"])
    best_fg_thresh_cl = np.mean(fg_threshs, axis=0)
    best_seed_thresh_cl = np.mean(seed_threshs, axis=0)
    print(best_fg_thresh_cl, best_seed_thresh_cl)
    return best_fg_thresh_cl, best_seed_thresh_cl


def center_crop(t, croph, cropw):
    h, w = t.shape[-2:]
    startw = w // 2 - (cropw // 2)
    starth = h // 2 - (croph // 2)
    return t[..., starth : starth + croph, startw : startw + cropw]


def prep_regression(gt_list, nclasses=6, class_names=CLASS_NAMES):
    # prepare gt regression
    cns = class_names[:nclasses]
    gt_regression = {}
    for i in range(nclasses):
        gt_regression[cns[i]] = []

    for gt in gt_list:
        gt_inst, gt_ct = gt[..., 0], gt[..., 1]
        ct_list = np.zeros(nclasses + 1)
        instance_map_tmp = center_crop(gt_inst, 224, 224)
        for instance in np.unique(instance_map_tmp):
            if instance == 0:
                continue
            ct_tmp = gt_ct[gt_inst == instance][0]
            if ct_tmp > nclasses:
                continue
            ct_list[int(ct_tmp)] += 1
        gt_reg = {}
        for i in range(nclasses):
            gt_reg[cns[i]] = ct_list[i + 1]

        for key in gt_regression.keys():
            gt_regression[key].append(gt_reg[key])

    for key in gt_regression.keys():
        gt_regression[key] = np.array(gt_regression[key])
    return gt_regression


def process_tile(
    i,
    pred_3c,
    pred_class,
    best_fg_thresh_cl,
    best_seed_thresh_cl,
    max_hole_size,
    min_threshs,
    max_threshs,
    nclasses,
    class_names,
):
    pred_inst, _ = make_instance_segmentation_cl(
        pred_3c,
        np.argmax(np.squeeze(pred_class)[1:], axis=0),
        fg_thresh_cl=best_fg_thresh_cl,
        seed_thresh_cl=best_seed_thresh_cl,
    )
    pred_inst = remove_holes_cv2(pred_inst, max_hole_size=max_hole_size)
    pred_inst = instance_wise_connected_components(pred_inst)
    ct_dict = make_ct(pred_class, pred_inst)
    pred_inst, pred_ct, ct_dict, ct_array, conic_crop = remove_obj_cls(
        pred_inst, ct_dict, min_threshs, max_threshs
    )
    pred_reg = convert_reg(ct_array[conic_crop], nclasses, class_names)
    return i, np.stack([pred_inst, pred_ct], axis=-1), pred_reg


def evaluate(
    pred_emb_list,
    pred_class_list,
    gt_regression,
    gt_list,
    best_fg_thresh_cl,
    best_seed_thresh_cl,
    params,
    criterium="lizard",
    nclasses=6,
    class_names=CLASS_NAMES,
    save_path=None,
    types=None,
):
    # set metrics to "" to skip metrics
    pred_list = []
    pred_regression = {}
    for i in range(nclasses):
        pred_regression[class_names[i]] = []

    # max_hole_size = 128 if pannuke else 50
    if params["dataset"] == "pannuke":
        min_threshs = MIN_THRESHS_PANNUKE if criterium == "all" else [0, 0, 0, 0, 0]
    else:
        min_threshs = BEST_MIN_THRESHS
    max_threshs = (
        MAX_THRESHS_PANNUKE if params["dataset"] == "pannuke" else BEST_MAX_THRESHS
    )

    res = []
    for i, (pred_3c, pred_class) in enumerate(zip(pred_emb_list, pred_class_list)):
        ri, pred_inst, pred_reg = process_tile(
            i,
            pred_3c,
            pred_class,
            best_fg_thresh_cl,
            best_seed_thresh_cl,
            params["max_hole_size"],
            min_threshs,
            max_threshs,
            nclasses,
            class_names,
        )
        res.append((pred_inst, pred_reg))
    for pred_inst, pred_reg in res:
        for key in pred_regression.keys():
            pred_regression[key].append(pred_reg[key])
        pred_list.append(pred_inst)

    for key in pred_regression.keys():
        pred_regression[key] = np.array(pred_regression[key])

    if criterium == "lizard":
        return lizard_eval(
            gt_list,
            pred_list,
            gt_regression,
            pred_regression,
            class_names,
            nclasses,
            save_path,
            criterium,
        )
    elif criterium == "f1":
        return alt_eval(
            params, gt_list, pred_list, class_names, nclasses, save_path, criterium
        )
    elif criterium == "pannuke":
        return pannuke_eval(gt_list, pred_list, types, criterium)
    elif criterium == "all":
        return (
            *lizard_eval(
                gt_list,
                pred_list,
                gt_regression,
                pred_regression,
                class_names,
                nclasses,
                save_path,
                criterium,
            ),
            pred_list,
            alt_eval(
                params,
                gt_list,
                pred_list,
                class_names,
                nclasses,
                save_path,
                criterium,
            ),
            *pannuke_eval(
                gt_list, pred_list, types, criterium, params["dataset"] == "lizard"
            ),
        )
    else:
        raise NotImplementedError("metric variation not implemented")


def lizard_eval(
    gt_list,
    pred_list,
    gt_regression,
    pred_regression,
    class_names,
    nclasses,
    save_path=None,
    criterium=None,
):
    df, mpq_list = calc_MPQ(pred_list, gt_list, nclasses)
    pq = df["pq"].values
    _, r2_list = get_multi_r2(gt_regression, pred_regression, class_names[:nclasses])
    if save_path is not None:
        with open(os.path.join(save_path[0], save_path[1] + "_pq.json"), "w") as js:
            json.dump(
                {
                    "pq": np.squeeze(df["pq"].values).tolist(),
                    "mpq": np.squeeze(df["multi_pq+"].values).tolist(),
                    "mpq_list": list(mpq_list),
                    "r2_list": list(r2_list),
                },
                js,
            )
    if criterium == "all":
        return mpq_list, r2_list, pq
    else:
        return {
            "optim": mpq_list,
            "r2": r2_list,
            "pq": pq,
        }


def pannuke_eval(gt_list, pred_list, types, criterium=None, skip=False):
    if skip:
        return (None, None, None)
    _, pan_bpq, pan_pq_list, pan_tiss = get_pannuke_pq(gt_list, pred_list, types)
    if criterium == "all":
        return pan_bpq, pan_pq_list, pan_tiss
    else:
        return {
            "optim": pan_pq_list,
            "bpq": pan_bpq,
            "tiss": pan_tiss,
        }


def alt_eval(
    params, gt_list, pred_list, class_names, nclasses, save_path, criterium=None
):
    ccrop = params["f1_metric_ccrop"]  # 256 if pannuke else 248
    metrics = per_tile_metrics(
        crop(gt_list, ccrop, ccrop),
        crop(np.stack(pred_list, axis=0), ccrop, ccrop),
        class_names[:nclasses],
        match_euc_dist=params["match_euc_dist"],  # 12 if pannuke else 6,
    )
    _, mdict = get_output(
        metrics,
        save_path,
        class_names[:nclasses],
    )
    if criterium == "all":
        return mdict
    else:
        return {
            "optim": pd.DataFrame(mdict["count_metrics"])
            .set_index("class")
            .loc[class_names[:nclasses], "F1"]
            .values,
            "hd": pd.DataFrame(mdict["class_wise_seg_metrics"])
            .set_index("class")
            .loc[class_names[:nclasses], "seg_hausdorff_(TP)"]
            .values,
        }


def make_instance_segmentation_cl(
    prediction, pred_semantic, fg_thresh_cl, seed_thresh_cl
):
    # prediction[0] = bg
    # prediction[1] = inside
    # prediction[2] = boundary
    ws_surface = 1.0 - prediction[1, ...]
    fg = np.zeros_like(ws_surface, dtype=bool)
    seeds = np.zeros_like(ws_surface, dtype=bool)
    for cl in range(len(fg_thresh_cl)):
        sem = pred_semantic == cl
        fg[sem] |= (1.0 - prediction[0][sem]) > fg_thresh_cl[cl]
        seeds[sem] |= prediction[1][sem] > seed_thresh_cl[cl]
    cnt, markers = cv2.connectedComponents((seeds > 0).astype(np.uint8), connectivity=8)
    labelling = watershed(ws_surface, markers, mask=fg).astype(np.uint16)
    return labelling, ws_surface


def remove_small_holescv2(img, sz):
    img = np.logical_not(img).astype(np.uint8)
    nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(img)
    sizes = stats[1:, -1]
    nb_blobs -= 1
    im_result = np.zeros((img.shape), dtype=np.uint16)
    for blob in range(nb_blobs):
        if sizes[blob] >= sz:
            im_result[im_with_separated_blobs == blob + 1] = 1

    im_result = np.logical_not(im_result)
    return im_result


def remove_holes_cv2(pred_inst, max_hole_size):
    out = np.zeros_like(pred_inst, dtype=np.uint16)
    for i in np.unique(pred_inst):
        if i == 0:
            continue
        out += remove_small_holescv2(pred_inst == i, max_hole_size) * i
    return out


def instance_wise_connected_components(pred_inst, connectivity=2):
    out = np.zeros_like(pred_inst)
    i = np.max(pred_inst) + 1
    for j in np.unique(pred_inst):
        if j == 0:
            continue
        nr_objects, relabeled = cv2.connectedComponents(
            (pred_inst == j).astype(np.uint8), connectivity=8
        )
        for new_lab in range(nr_objects):
            if new_lab == 0:
                continue
            out[relabeled == new_lab] = i
            i += 1
    return out


def make_ct(pred_class, instance_map):
    slices = find_objects(instance_map)
    pred_class = np.rollaxis(pred_class, 0, 3)
    # pred_class = softmax(pred_class,0)
    out = []
    out.append((0, 0))
    for i, sl in enumerate(slices):
        i += 1
        if sl:
            inst = instance_map[sl] == i
            i_cls = softmax(pred_class[sl])[inst]
            i_cls = np.sum(i_cls, axis=0)[1:].argmax() + 1
            out.append((i, i_cls))
    out_ = np.array(out)
    pred_ct = {str(k): int(v) for k, v in out_ if v != 0}
    return pred_ct


def is_in_center(sl, conic_dummy):
    return np.count_nonzero(conic_dummy[sl]) > 0


def remove_obj_cls(
    pred_inst, pred_cls_dict, min_threshs=BEST_MIN_THRESHS, max_threshs=BEST_MAX_THRESHS
):
    out_oi = np.zeros_like(pred_inst, dtype=np.int64)
    i_ = 1
    out_oc = []
    out_oc.append((0, 0))
    slices = find_objects(pred_inst)
    # special treatment for conic r2 which is only counted in the center 224x224 crop
    conic_dummy = np.zeros_like(pred_inst)
    conic_dummy[16:-16, 16:-16] = 1
    conic_crop = [False]

    for i, sl in enumerate(slices):
        if sl:
            i += 1
            px = np.count_nonzero(pred_inst[sl] == i)
            cls_ = pred_cls_dict[str(i)]
            if (px > min_threshs[cls_ - 1]) & (px < max_threshs[cls_ - 1]):
                out_oc.append((i_, cls_))
                out_oi[sl][pred_inst[sl] == i] = i_
                i_ += 1
                if is_in_center(sl, conic_dummy):
                    conic_crop.append(True)
                else:
                    conic_crop.append(False)

    out_oc = np.array(out_oc)
    out_dict = {str(k): int(v) for k, v in out_oc if v != 0}
    pcls_list = np.array([0] + list(out_dict.values()))
    pcls_keys = np.array(["0"] + list(out_dict.keys())).astype(int)
    lookup = np.zeros(pcls_keys.max() + 1)
    lookup[pcls_keys] = pcls_list
    cls_map = lookup[out_oi]
    return out_oi, cls_map, out_dict, out_oc, np.array(conic_crop)


def convert_reg(out_oc, nclasses=6, class_names=CLASS_NAMES):
    pred_regression = {}
    for i in range(nclasses):
        pred_regression[class_names[i]] = np.count_nonzero(out_oc[:, 1] == (i + 1))
    return pred_regression
