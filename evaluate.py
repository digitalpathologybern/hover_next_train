import os
import toml
import torch
from torch.utils.data import DataLoader
import numpy as np
import gc
import pandas as pd
import json
import argparse

from src.multi_head_unet import get_model, load_checkpoint
from src.inference_utils import run_inference
from src.post_proc_utils import prep_regression, evaluate, get_pp_params
from src.spatial_augmenter import SpatialAugmenter
from src.data_utils import SliceDataset
from src.constants import CLASS_NAMES, CLASS_NAMES_PANNUKE, PANNUKE_FOLDS
from src.color_conversion import color_augmentations  # , get_normalize


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


def process_and_save(res, out_p, dsname, tta, class_names=CLASS_NAMES):
    # def process_and_save(res, out_p, dsname):
    # [mpq_list, r2_list, mdict, pq, pan_bpq, pan_pq_list, pan_tiss]
    mean_dict = {}
    std_dict = {}
    nrounds = len(res)
    if res[0][0] is not None:
        mpq_m = np.mean([r[0] for r in res], axis=0)
        mpq_s = np.std([r[0] for r in res], axis=0)
        pq_m = np.mean([r[3] for r in res], axis=0)
        pq_s = np.std([r[3] for r in res], axis=0)

    if res[0][1] is not None:
        r2 = [[r_ if r_ >= 0 else 0 for r_ in r[1]] for r in res]
        r2_m = np.mean(r2, axis=0)
        r2_s = np.std(r2, axis=0)

    if res[0][2] is not None:
        mdict = [r[2] for r in res]
        mean_dict |= mdict[0].copy()
        std_dict |= mdict[0].copy()
        for k in mdict[0].keys():
            if k == "binary_pixel_metrics":
                pixel_f1_mean = np.mean([r[k][0]["mean"] for r in mdict], axis=0)
                pixel_f1_std = np.std([r[k][0]["mean"] for r in mdict], axis=0)
                pixel_mcc_mean = np.mean([r[k][1]["mean"] for r in mdict], axis=0)
                pixel_mcc_std = np.std([r[k][1]["mean"] for r in mdict], axis=0)
                mean_dict[k] = {"pixel_f1": pixel_f1_mean, "pixel_mcc": pixel_mcc_mean}
                std_dict[k] = {"pixel_f1": pixel_f1_std, "pixel_mcc": pixel_mcc_std}
            else:
                cm = pd.DataFrame(mean_dict[k])
                cs = pd.DataFrame(std_dict[k])
                cm.iloc[:, 1:] = np.mean(
                    [
                        pd.DataFrame(r[k]).iloc[:, 1:].values.astype(float)
                        for r in mdict
                    ],
                    axis=0,
                )
                cs.iloc[:, 1:] = np.std(
                    [
                        pd.DataFrame(r[k]).iloc[:, 1:].values.astype(float)
                        for r in mdict
                    ],
                    axis=0,
                )

                cm.replace(np.nan, -999, inplace=True)
                cs.replace(np.nan, -999, inplace=True)
                mean_dict[k] = cm.to_dict("records")
                std_dict[k] = cs.to_dict("records")
    if res[0][4] is not None:
        pannuke_bpq_mean = np.nanmean([r[4] for r in res])
        pannuke_bpq_std = np.nanstd([r[4] for r in res])
        pannuke_mpq_mean = np.nanmean([r[5] for r in res], axis=0)
        pannuke_mpq_std = np.nanstd([r[5] for r in res], axis=0)
        mean_dict["pannuke_metrics"] = {
            "bpq": pannuke_bpq_mean,
            "mpq": pannuke_mpq_mean.tolist(),
        }
        std_dict["pannuke_metrics"] = {
            "bpq": pannuke_bpq_std,
            "mpq": pannuke_mpq_std.tolist(),
        }
    if res[0][6] is not None:
        tiss_mpq = []
        tiss_bpq = []
        for r in res:
            tiss_mpq_, tiss_bpq_ = r[6]
            tiss_mpq.append(list(tiss_mpq_.values()))
            tiss_bpq.append(list(tiss_bpq_.values()))
        tiss_mpq_mean = {
            k: v for k, v in zip(tiss_mpq_.keys(), np.nanmean(tiss_mpq, axis=0))
        }
        tiss_mpq_std = {
            k: v for k, v in zip(tiss_mpq_.keys(), np.nanstd(tiss_mpq, axis=0))
        }
        tiss_bpq_mean = {
            k: v for k, v in zip(tiss_bpq_.keys(), np.nanmean(tiss_bpq, axis=0))
        }
        tiss_bpq_std = {
            k: v for k, v in zip(tiss_bpq_.keys(), np.nanstd(tiss_bpq, axis=0))
        }
        mean_dict["pannuke_metrics"] |= {
            "tiss_mpq": tiss_mpq_mean,
            "tiss_bpq": tiss_bpq_mean,
        }
        std_dict["pannuke_metrics"] |= {
            "tiss_mpq": tiss_mpq_std,
            "tiss_bpq": tiss_bpq_std,
        }

    if res[0][0] is not None:
        mean_dict["old_metrics"] = [
            {"class": k, "mpq": mpq, "r2": r2}
            for k, mpq, r2 in zip(class_names, mpq_m.tolist(), r2_m.tolist())
        ]
        mean_dict["old_metrics"].append(
            {"class": "all", "mpq": pq_m.tolist()[0], "r2": 0}
        )
        std_dict["old_metrics"] = [
            {"class": k, "mpq": mpq, "r2": r2}
            for k, mpq, r2 in zip(class_names, mpq_s.tolist(), r2_s.tolist())
        ]
        std_dict["old_metrics"].append(
            {"class": "all", "mpq": pq_s.tolist()[0], "r2": 0}
        )
    print(
        "saving to",
        os.path.join(out_p, f"{dsname}_mean_metrics_tta_{tta}_n_{nrounds}.json"),
    )
    with open(
        os.path.join(out_p, f"{dsname}_mean_metrics_tta_{tta}_n_{nrounds}.json"), "w"
    ) as f:
        json.dump(mean_dict, f)
    with open(
        os.path.join(out_p, f"{dsname}_std_metrics_tta_{tta}_n_{nrounds}.json"), "w"
    ) as f:
        json.dump(std_dict, f)


def evaluate_tile_dataset(
    ds,
    models,
    dsname,
    experiments,
    params,
    nclasses=7,
    class_names=CLASS_NAMES,
    rank=0,
    types=None,
):
    color_aug_fn = color_augmentations(False, s=0.2, rank=rank)
    aug = SpatialAugmenter(aug_params_slow)
    # normalization = get_normalize(use_norm=params["dataset"] == "pannuke")
    data_loader = DataLoader(
        ds,
        batch_size=params["validation_batch_size"],
        shuffle=False,
        prefetch_factor=4,
        num_workers=params["num_workers"],
    )
    out_p = os.path.join(params["experiment"], params["eval_optim_metric"])
    if not os.path.exists(out_p):
        os.makedirs(out_p)
    best_fg_thresh_cl, best_seed_thresh_cl = get_pp_params(
        experiments, "", True, eval_metric=params["eval_optim_metric"]
    )

    res = []

    for i in range(params["n_rounds"]):
        print(f"round {i}")

        pred_emb_list, pred_class_list, gt_list, raw_list = run_inference(
            data_loader,
            models,
            aug,
            color_aug_fn,
            tta=params["tta"],
            rank=rank,
        )
        if i == 0:
            gt_regression = prep_regression(
                gt_list, nclasses=nclasses, class_names=class_names
            )
        (
            mpq_list,
            r2_list,
            pq,
            pred_list,
            mdict,
            pan_bpq,
            pan_pq_list,
            pan_tiss,
        ) = evaluate(
            pred_emb_list,
            pred_class_list,
            gt_regression,
            gt_list,
            best_fg_thresh_cl,
            best_seed_thresh_cl,
            params,
            "all",
            nclasses,
            class_names,
            types=types,
        )

        if params["save"]:
            np.save(
                os.path.join(out_p, dsname + f"_r{i}_" + ".npy"),
                np.stack(pred_list),
            )
        res.append([mpq_list, r2_list, mdict, pq, pan_bpq, pan_pq_list, pan_tiss])
    process_and_save(res, out_p, dsname, tta=params["tta"], class_names=class_names)


def main(nclasses, class_names, cp_paths, params, rank=0):
    print("main")
    # load data and create slice_dataset
    ds_list = []
    ds_names = []
    fold = params["fold"]
    types_fold = None
    if params["dataset"] == "pannuke":
        _, test_f = PANNUKE_FOLDS[fold - 1]
        i = test_f + 1
        raw_fold = np.load(
            os.path.join(params["data_path"], "images", "fold" + str(i), "images.npy"),
            mmap_mode="r",
        )
        gt_fold = np.load(
            os.path.join(params["data_path"], "masks", "fold" + str(i), "labels.npy"),
            mmap_mode="r",
        )
        types_fold = np.load(
            os.path.join(params["data_path"], "images", "fold" + str(i), "types.npy"),
            mmap_mode="r",
        )
        ds_list.append(SliceDataset(raw=raw_fold, labels=gt_fold))
        ds_names.append("pannuke_test")
    else:
        # load complete lizard
        liz_test_ds = SliceDataset(
            raw=np.load(
                os.path.join(params["data_path_liz"], "test_images.npy"), mmap_mode="r"
            ),
            labels=np.load(
                os.path.join(params["data_path_liz"], "test_labels.npy"), mmap_mode="r"
            ),
        )
        ds_list.extend([liz_test_ds])
        ds_names.extend(["lizard_test"])

        # load mitosis
        mit_test_ds = SliceDataset(
            raw=np.load(
                os.path.join(params["data_path_mit"], "test_ds", "test_img.npy"),
                mmap_mode="r",
            ),
            labels=np.load(
                os.path.join(params["data_path_mit"], "test_ds", "test_lab.npy"),
                mmap_mode="r",
            ),
        )
        ds_list.extend([mit_test_ds])
        ds_names.extend(["mitosis_test"])

    # load models
    models = []
    for pth in cp_paths:
        checkpoint_path = f"{pth}/train/best_model"
        print(checkpoint_path)
        enc = params["encoder"]
        model = get_model(
            enc=enc, out_channels_cls=nclasses + 1, out_channels_inst=5
        ).to(rank)
        model, _, _ = load_checkpoint(model, checkpoint_path, 0)
        model.eval()
        models.append(model)
    for ds, dsname in zip(
        ds_list,
        ds_names,
    ):  # liz_ds "lizard",
        if ds is None:
            continue
        evaluate_tile_dataset(
            ds,
            models,
            dsname,
            cp_paths,
            params,
            nclasses,
            class_names,
            rank,
            types=types_fold,
        )
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        help="experiment name, specify with fold e.g. test_experiment_1, "
        + "can be done with ensembles e.g, by specifying:"
        + " --exp test_experiment_1,test_experiment_2,test_experiment_3",
    )
    parser.add_argument(
        "--tta",
        type=int,
        default=16,
        help="number of test time augmentation views",
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=5,
        help="average over n rounds",
    )
    

    args = parser.parse_args()
    params = toml.load(f"{args.exp}/params.toml")
    params["experiment"] = "_".join(args.exp.split(","))
    params["tta"] = int(args.tta)
    if params["tta"] <= 0:
        params["n_rounds"] = 1
    else:
        params["n_rounds"] = int(args.n_rounds)
    class_names = CLASS_NAMES_PANNUKE if params["dataset"] == "pannuke" else CLASS_NAMES
    rank = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nclasses = 5 if params["dataset"] == "pannuke" else 7
    main(nclasses, class_names, args.exp.split(","), params, rank)
