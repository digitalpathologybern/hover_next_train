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

from src.color_conversion import color_augmentations  # , get_normalize

import json
from tqdm.auto import tqdm
import pandas as pd
import argparse

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)

aug_params_slow = {
    "mirror": {"prob_x": 0.5, "prob_y": 0.5, "prob": 0.85},
    "translate": {"max_percent": 0.05, "prob": 0.0},
    "scale": {"min": 0.8, "max": 1.2, "prob": 0.0},
    "zoom": {"min": 0.8, "max": 1.2, "prob": 0.0},
    "rotate": {"rot90": True, "prob": 0.85},
    "shear": {"max_percent": 0.1, "prob": 0.0},
    "elastic": {"alpha": [120, 120], "sigma": 8, "prob": 0.0},
}


def find_hyperparameters(
    ds, models, name, nclasses=7, class_names=CLASS_NAMES, rank=0, random_seed=42
):
    color_aug_fn = color_augmentations(False, s=0.2, rank=rank)
    # normalization = get_normalize(use_norm=params["dataset"] == "pannuke")
    aug = SpatialAugmenter(aug_params_slow, random_seed=random_seed)
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

    out_dict = {}
    if params["eval_criteria"] != "":
        for criterium in params["eval_criteria"].split("|"):
            print(criterium, "| searching best fg threshold", flush=True)
            best_seed_thresh_cl = [0.3] * nclasses
            optim_list_global = []
            fg_threshs = np.linspace(0.1, 0.9, 9)
            for fg_thresh in fg_threshs:
                print("FG:", fg_thresh, flush=True)
                eval_dict = evaluate(
                    pred_emb_list,
                    pred_class_list,
                    gt_regression,
                    gt_list,
                    [fg_thresh] * nclasses,
                    best_seed_thresh_cl,
                    params,
                    criterium,
                    nclasses,
                    class_names,
                )
                optim_list_global.append(eval_dict["optim"])

            best_idx = np.stack(optim_list_global)
            best_fg_thresh_cl = [fg_threshs[i] for i in best_idx.argmax(0)]
            out_dict[f"best_fg_{criterium}"] = best_fg_thresh_cl
            optim_list_global = []

            print(criterium, "| searching best seed threshold", flush=True)

            seed_threshs = np.linspace(0.1, 0.9, 9)
            for seed_thresh in seed_threshs:
                print("Seed:", seed_thresh, flush=True)
                eval_dict = evaluate(
                    pred_emb_list,
                    pred_class_list,
                    gt_regression,
                    gt_list,
                    best_fg_thresh_cl,
                    [seed_thresh] * nclasses,
                    params,
                    criterium,
                    nclasses,
                    class_names,
                )
                optim_list_global.append(eval_dict["optim"])
            best_idx = np.stack(optim_list_global)
            best_seed_thresh_cl = [seed_threshs[i] for i in best_idx.argmax(0)]
            out_dict[f"best_seed_{criterium}"] = best_seed_thresh_cl
            print(criterium)
            print(best_fg_thresh_cl)
            print(best_seed_thresh_cl)

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
        _, test_f = PANNUKE_FOLDS[int(params["fold"]) - 1]
        i = test_f + 1
        raw_fold = np.load(
            os.path.join(params["data_path"], "images", "fold" + str(i), "images.npy"),
            mmap_mode="r",
        )
        gt_fold = np.load(
            os.path.join(params["data_path"], "masks", "fold" + str(i), "labels.npy"),
            mmap_mode="r",
        )
        ds_list = [SliceDataset(raw=raw_fold, labels=gt_fold)]
        ds_names = ["pannuke_test"]
        class_names = CLASS_NAMES_PANNUKE
    else:
        # Mitosis dataset test set (real annotations)
        x_mit_test = np.load(os.path.join(params["data_path_mit"], "test_ds/test_img.npy"))
        y_mit_test = np.load(os.path.join(params["data_path_mit"], "test_ds/test_lab.npy"))

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
            ds,
            [model],
            name,
            nclasses=nclasses,
            class_names=class_names,
            rank=rank,
            random_seed=params["seed"],
        )
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to .toml file of experiment. e.g. lizard_exp_1/params.toml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model",
        help="checkpoint to load. e.g. best_model, checkpoint_step_10000. Use this to evaluate other checkpoints",
    )
    args = parser.parse_args()
    params = toml.load(args.config)
    print(
        "loaded config for",
        params["experiment"],
        "\n starting hyperparameter search...",
        flush=True,
    )
    params["checkpoint_path"] = args.checkpoint
    rank = torch.cuda.current_device()
    nclasses = 5 if params["dataset"] == "pannuke" else 7
    main(nclasses, params, rank)
