import os
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.multi_head_unet import get_model, load_checkpoint
from src.inference_utils import run_inference
from src.post_proc_utils import (
    prep_regression,
    evaluate,
)
from src.constants import CLASS_NAMES, CLASS_NAMES_PANNUKE
import toml

from src.spatial_augmenter import SpatialAugmenter
from src.data_utils import SliceDataset, PANNUKE_FOLDS

from src.color_conversion import color_augmentations

import json
from tqdm.auto import tqdm
import pandas as pd
import argparse

torch.backends.cudnn.benchmark = True
torch.manual_seed(420)

aug_params_slow = {
    "mirror": {"prob_x": 0.5, "prob_y": 0.5, "prob": 0.85},
    "translate": {"max_percent": 0.05, "prob": 0.0},
    "scale": {"min": 0.8, "max": 1.2, "prob": 0.0},
    "zoom": {"min": 0.8, "max": 1.2, "prob": 0.0},
    "rotate": {"rot90": True, "prob": 0.85},
    "shear": {"max_percent": 0.1, "prob": 0.0},
    "elastic": {"alpha": [120, 120], "sigma": 8, "prob": 0.0},
}


def find_hyperparameters(ds, models, name, nclasses=7, class_names=CLASS_NAMES, rank=0):
    color_aug_fn = color_augmentations(False, s=0.2, rank=rank)
    aug = SpatialAugmenter(aug_params_slow)
    data_loader = DataLoader(
        ds,
        batch_size=params["validation_batch_size"],
        shuffle=False,
        prefetch_factor=4,
        num_workers=params["num_workers"],
    )

    pred_emb_list, pred_class_list, gt_list, _ = run_inference(
        data_loader, models, aug, color_aug_fn, params["tta"], rank=rank
    )
    gt_regression = prep_regression(gt_list, nclasses=nclasses, class_names=class_names)
    print("searching best fg threshold")

    if params["pannuke"]:
        best_fg_thresh_cl = [0.5] * nclasses
        best_seed_thresh_cl = [0.3] * nclasses
    else:
        best_fg_thresh_cl = [0.47, 0.57, 0.62, 0.64, 0.52, 0.49, 0.55]
        best_seed_thresh_cl = [0.24, 0.48, 0.42, 0.58, 0.72, 0.28, 0.42]

    mpq_list_global = []
    r2_list_global = []
    f1_list_global = []
    hd_list_global = []
    fg_threshs = np.linspace(0.1, 0.9, 9)
    for fg_thresh in fg_threshs:
        print("FG:", fg_thresh)
        (
            mpq_list,
            r2_list,
            pred_list,
            pred_regression,
            mdict,
            pq,
            pan_bpq,
            pan_pq_list,
            pan_tiss,
        ) = evaluate(
            pred_emb_list,
            pred_class_list,
            gt_regression,
            gt_list,
            [fg_thresh] * nclasses,
            best_seed_thresh_cl,
            params,
            nclasses,
            class_names,
        )
        mpq_list_global.append(mpq_list)
        r2_list_global.append(r2_list)
        f1_list_global.append(
            pd.DataFrame(mdict["count_metrics"])
            .set_index("class")
            .loc[class_names[:nclasses], "F1"]
            .values
        )
        hd_list_global.append(
            pd.DataFrame(mdict["class_wise_seg_metrics"])
            .set_index("class")
            .loc[class_names[:nclasses], "seg_hausdorff_(TP)"]
            .values
        )

    # mpq
    best_idx = np.stack(mpq_list_global)
    best_fg_thresh_cl = [fg_threshs[i] for i in best_idx.argmax(0)]
    # r2
    best_idx_r2 = np.stack(r2_list_global)
    best_fg_thresh_cl_r2 = [fg_threshs[i] for i in best_idx_r2.argmax(0)]

    # f1
    best_idx_f1 = np.stack(f1_list_global)
    best_fg_thresh_cl_f1 = [fg_threshs[i] for i in best_idx_f1.argmax(0)]
    # hd
    best_idx_hd = np.stack(hd_list_global)
    best_fg_thresh_cl_hd = [fg_threshs[i] for i in best_idx_hd.argmin(0)]

    out_dict = {
        "best_fg_mpq": best_fg_thresh_cl,
        "best_fg_r2": best_fg_thresh_cl_r2,
        "best_fg_f1": best_fg_thresh_cl_f1,
        "best_fg_hd": best_fg_thresh_cl_hd,
    }
    for c, i in enumerate(fg_threshs):
        out_dict[f"fg_{i}_mpq"] = best_idx[c].tolist()
        out_dict[f"fg_{i}_r2"] = best_idx_r2[c].tolist()
        out_dict[f"fg_{i}_f1"] = best_idx_f1[c].tolist()
        out_dict[f"fg_{i}_hd"] = best_idx_hd[c].tolist()

    print("searching best seed threshold")

    mpq_list_global = []
    r2_list_global = []
    f1_list_global = []
    hd_list_global = []
    # seed_threshs = np.linspace(0.2,0.8,61)

    seed_threshs = np.linspace(0.1, 0.9, 9)
    for seed_thresh in seed_threshs:
        seed_thresh = [seed_thresh] * nclasses
        (
            mpq_list,
            r2_list,
            pred_list,
            pred_regression,
            mdict,
            pq,
            pan_bpq,
            pan_pq_list,
            pan_tiss,
        ) = evaluate(
            pred_emb_list,
            pred_class_list,
            gt_regression,
            gt_list,
            best_fg_thresh_cl,
            seed_thresh,
            params,
            nclasses,
            class_names,
        )
        mpq_list_global.append(mpq_list)
        r2_list_global.append(r2_list)
        f1_list_global.append(
            pd.DataFrame(mdict["count_metrics"])
            .set_index("class")
            .loc[class_names[:nclasses], "F1"]
            .values
        )
        hd_list_global.append(
            pd.DataFrame(mdict["class_wise_seg_metrics"])
            .set_index("class")
            .loc[class_names[:nclasses], "seg_hausdorff_(TP)"]
            .values
        )

    best_idx = np.stack(mpq_list_global)
    best_seed_thresh_cl = [seed_threshs[i] for i in best_idx.argmax(0)]

    # r2
    best_idx_r2 = np.stack(r2_list_global)
    best_seed_thresh_cl_r2 = [seed_threshs[i] for i in best_idx_r2.argmax(0)]
    # f1
    best_idx_f1 = np.stack(f1_list_global)
    best_seed_cl_f1 = [seed_threshs[i] for i in best_idx_f1.argmax(0)]
    # hd
    best_idx_hd = np.stack(hd_list_global)
    best_seed_cl_hd = [seed_threshs[i] for i in best_idx_hd.argmin(0)]

    # calculate best mPQ and r2
    mpq_list_global_stack = np.stack(mpq_list_global)
    best_mpq = np.array(
        [
            mpq[i]
            for i, mpq in zip(best_idx.argmax(0), np.transpose(mpq_list_global_stack))
        ]
    )

    r2_list_global_stack = np.stack(r2_list_global)
    best_r2 = np.array(
        [
            r2[i]
            for i, r2 in zip(best_idx_r2.argmax(0), np.transpose(r2_list_global_stack))
        ]
    )

    f1_list_global_stack = np.stack(f1_list_global)
    best_f1 = np.array(
        [
            f1[i]
            for i, f1 in zip(best_idx_f1.argmax(0), np.transpose(f1_list_global_stack))
        ]
    )

    hd_list_global_stack = np.stack(hd_list_global)
    best_hd = np.array(
        [
            hd[i]
            for i, hd in zip(best_idx_hd.argmin(0), np.transpose(hd_list_global_stack))
        ]
    )

    out_dict["best_seed_mpq"] = best_seed_thresh_cl
    out_dict["best_seed_r2"] = best_seed_thresh_cl_r2
    out_dict["best_seed_f1"] = best_seed_cl_f1
    out_dict["best_seed_hd"] = best_seed_cl_hd

    out_dict["best_r2_overall"] = np.mean(best_r2)
    out_dict["best_mpq_overall"] = np.mean(best_mpq)
    out_dict["best_f1_overall"] = np.mean(best_f1)
    out_dict["best_hd_overall"] = np.mean(best_hd)

    for c, i in enumerate(seed_threshs):
        out_dict[f"seed_{i}_mpq"] = best_idx[c].tolist()
        out_dict[f"seed_{i}_r2"] = best_idx_r2[c].tolist()
        out_dict[f"seed_{i}_f1"] = best_idx_f1[c].tolist()
        out_dict[f"seed_{i}_hd"] = best_idx_hd[c].tolist()

    print("r2")
    print(best_seed_thresh_cl_r2)
    print(best_fg_thresh_cl_r2)
    print("mpq")
    print(best_seed_thresh_cl)
    print(best_fg_thresh_cl)
    print("f1")
    print(best_seed_cl_f1)
    print(best_fg_thresh_cl_f1)
    print("hd")
    print(best_seed_cl_hd)
    print(best_fg_thresh_cl_hd)

    with open(os.path.join(params["experiment"], name + "_param_dict.json"), "w") as f:
        json.dump(out_dict, f)


def main(nclasses, params, rank=0):
    # load model
    model = get_model(
        enc=params["encoder"],
        out_channels_cls=params["out_channels_cls"],
        out_channels_inst=params["inst_channels"],
    ).to(rank)
    cp_path = os.path.join(params["experiment"], "train", params["checkpoint_path"])
    model, _, _ = load_checkpoint(model, cp_path, rank=0)
    model.eval()
    if params["dataset"] == "pannuke":
        _, test_f = PANNUKE_FOLDS[fold - 1]
        i = test_f + 1
        raw_fold = np.load(
            os.path.join(
                params["data_path_pannuke"], "images", "fold" + str(i), "images.npy"
            ),
            mmap_mode="r",
        )
        gt_fold = np.load(
            os.path.join(
                params["data_path_pannuke"], "masks", "fold" + str(i), "labels.npy"
            ),
            mmap_mode="r",
        )
        ds_list = [SliceDataset(raw=raw_fold, labels=gt_fold)]
        ds_names = ["pannuke_test"]
        class_names = CLASS_NAMES_PANNUKE
    else:
        # Mitosis dataset test set (real annotations)
        x_mit_test = np.load(os.path.join(params["data_path_mit"], "test_img.npy"))
        y_mit_test = np.load(os.path.join(params["data_path_mit"], "test_lab.npy"))

        mit_test_ds = SliceDataset(raw=x_mit_test, labels=y_mit_test)

        # Lizard dataset test set
        x_liz_test = np.load(os.path.join(params["data_path_liz"], "test_images.npy"))
        y_liz_test = np.load(os.path.join(params["data_path_liz"], "test_labels.npy"))

        liz_test_ds = SliceDataset(raw=x_liz_test, labels=y_liz_test)
        ds_list = [mit_test_ds, liz_test_ds]
        ds_names = ["mit_test", "liz_test"]
        class_names = CLASS_NAMES
    print("evaluating for ", class_names, "on", ds_names)
    for ds, name in zip(ds_list, ds_names):
        find_hyperparameters(
            ds, [model], name, nclasses=nclasses, class_names=class_names, rank=rank
        )
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-exp",
        type=str,
        default=None,
        help="experiment name, specify with fold e.g. test_experiment_1",
    )
    args = parser.parse_args()
    params = toml.load(f"{args.exp}/params.toml")
    fold = int(params["fold"])

    rank = 0  # ignoring that you might not want to use gpu:0 or cpu instead :)
    nclasses = 5 if params["dataset"] == "pannuke" else 7
    main(nclasses, params, rank)
