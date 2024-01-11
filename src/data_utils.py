from torch.utils.data import Dataset
import numpy as np
import torch
from typing import Optional, List, Tuple, Callable
from torch.utils.data import Dataset
from tqdm import tqdm
import mahotas as mh
import os
from torch.utils.data import DataLoader, Dataset

from torch.utils.data.distributed import DistributedSampler
from src.constants import (
    PANNUKE_FOLDS,
    CLASS_NAMES,
    CLASS_NAMES_PANNUKE,
)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_cpvs(gt_inst):
    # only works for batchsize = 1 and 2d
    # using mean instead of median because its faster
    device = gt_inst.device
    gt_inst = gt_inst.squeeze()  # H x W or H x W x D
    cpvs = torch.zeros((2,) + gt_inst.shape, dtype=torch.float).to(device)
    ind_x, ind_y = gt_inst.nonzero(as_tuple=True)
    val = gt_inst[ind_x, ind_y]
    labels = val.unique()
    for label in labels:
        sel = val == label
        x = ind_x[sel]
        y = ind_y[sel]
        # x, y = ind_x[sel], ind_y[sel]#(gt_inst == label).long().nonzero(as_tuple=True)
        cpvs[0, x, y] = -x + x.float().mean()
        cpvs[1, x, y] = -y + y.float().mean()
    return cpvs.unsqueeze(0)


@torch.jit.script
def jit_cpvs(gt_inst):
    # only works for batchsize = 1 and 2d
    # using mean instead of median because its faster
    device = gt_inst.device
    gt_inst = gt_inst.squeeze()  # H x W or H x W x D
    cpvs = torch.zeros((2,) + gt_inst.shape, dtype=torch.float, device=device)
    ind = gt_inst.nonzero().T
    val = gt_inst[ind[0], ind[1]]
    labels = torch.unique(val)
    for label in labels:
        sel = val == label
        x = ind[0][sel]
        y = ind[1][sel]
        # x, y = ind_x[sel], ind_y[sel]#(gt_inst == label).long().nonzero(as_tuple=True)
        cpvs[0, x, y] = -x + x.float().mean()
        cpvs[1, x, y] = -y + y.float().mean()
    return cpvs.unsqueeze(0)


@torch.jit.script
def parallel_cpvs(gt_inst):
    futures: List[torch.jit.Future[torch.Tensor]] = []
    for i in range(gt_inst.shape[0]):
        futures.append(torch.jit.fork(jit_cpvs, gt_inst[i]))
    results = []
    for future in futures:
        results.append(torch.jit.wait(future))

    return torch.cat(results, dim=0)


def normalize_percentile(
    x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-8, dtype=np.float32
):
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_min_max(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_min_max(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if mi is None:
        mi = np.min(x)
    if ma is None:
        ma = np.max(x)
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)
    return x


class SliceDataset(Dataset):
    def __init__(self, raw, labels, norm=True):
        self.raw = raw
        self.labels = labels
        self.norm = norm

    def __len__(self):
        return self.raw.shape[0]

    def __getitem__(self, idx):
        raw_tmp = self.raw[idx].astype(np.float32)
        if self.norm:
            raw_tmp = normalize_min_max(raw_tmp, 0, 255)
        if self.labels is not None:
            return raw_tmp, self.labels[idx].astype(np.float32)
        else:
            return raw_tmp, False


class GaussianNoise(torch.nn.Module):
    def __init__(self, sigma, rank):
        super().__init__()
        self.sigma = sigma
        self.rank = rank

    def forward(self, img):
        noise = torch.randn(img.shape).to(self.rank) * self.sigma
        return img + noise


def center_crop(t, croph, cropw):
    h, w = t.shape[-2:]
    startw = w // 2 - (cropw // 2)
    starth = h // 2 - (croph // 2)
    return t[..., starth : starth + croph, startw : startw + cropw]


def inst_to_3c(gt_labels):
    borders = mh.labeled.borders(gt_labels, Bc=np.ones((3, 3)))
    mask = gt_labels > 0
    return (((borders & mask) * 1) + (mask * 1))[np.newaxis, :]


def add_3c_gt_fast(Y):
    print("adding 3-class ground truth...")
    instances = Y[..., 0]
    gt_3c_list = []
    for inst in tqdm(instances):
        gt_3c = inst_to_3c(inst)
        gt_3c_list.append(gt_3c)
    gt_3c = np.transpose(np.stack(gt_3c_list, 0), [0, 2, 3, 1])
    Y = np.concatenate([Y, gt_3c], -1)
    return Y


def get_pannuke(params):
    fold = params["fold"] - 1
    im_folds = [
        np.load(
            os.path.join(params["data_path"], "images", "fold" + str(i), "images.npy"),
            mmap_mode="r",
        )
        for i in range(1, 4)
    ]
    im_types = [
        np.load(
            os.path.join(params["data_path"], "images", "fold" + str(i), "types.npy"),
            mmap_mode="r",
        )
        for i in range(1, 4)
    ]
    gt_folds = [
        np.load(
            os.path.join(params["data_path"], "masks", "fold" + str(i), "labels.npy"),
            mmap_mode="r",
        )
        for i in range(1, 4)
    ]
    val_f, test_f = PANNUKE_FOLDS[fold]
    if params["test_as_val"]:
        x_train = np.concatenate([im_folds[fold], im_folds[val_f]])
        train_types = np.concatenate([im_types[fold], im_types[val_f]])
        y_train = np.concatenate([gt_folds[fold], gt_folds[val_f]])
        x_val = im_folds[test_f]
        y_val = gt_folds[test_f]
    else:
        x_train = im_folds[fold]
        y_train = gt_folds[fold]
        train_types = im_types[fold]
        x_val = im_folds[val_f]
        y_val = gt_folds[val_f]

    labeled_dataset = SliceDataset(raw=x_train, labels=add_3c_gt_fast(y_train))
    validation_dataset = SliceDataset(raw=x_val, labels=add_3c_gt_fast(y_val))

    labeled_dataloader = DataLoader(
        labeled_dataset,
        batch_size=params["batch_size"],
        prefetch_factor=2,
        sampler=get_weighted_sampler(labeled_dataset, [0, 1, 2, 3, 4, 5]),
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    dist_samp = DistributedSampler(
        validation_dataset,
        shuffle=True,
        drop_last=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        sampler=dist_samp,
        batch_size=params["validation_batch_size"],
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    sz = int(x_train.shape[0] / params["batch_size"])

    return (
        [labeled_dataloader],
        validation_dataloader,
        sz,
        dist_samp,
        CLASS_NAMES_PANNUKE,
    )


def get_data(params):
    if params["dataset"] == "pannuke":
        return get_pannuke(params)
    elif params["dataset"] == "lizard":
        return get_lizard(params)
    else:
        raise NotImplementedError(
            "dataset not included, choose from 'pannuke' or 'lizard'"
        )


def get_lizard(params):
    fold_path = os.path.join(params["data_path_liz"], "fold_" + str(params["fold"]))
    if params["test_as_val"]:
        Liz_X_train = np.concatenate(
            [
                np.load(
                    os.path.join(
                        fold_path,
                        "train_img.npy",
                    )
                ),
                np.load(
                    os.path.join(
                        fold_path,
                        "valid_img.npy",
                    )
                ),
            ]
        )
        Liz_Y_train = add_3c_gt_fast(
            np.concatenate(
                [
                    np.load(
                        os.path.join(
                            fold_path,
                            "train_lab.npy",
                        )
                    ),
                    np.load(
                        os.path.join(
                            fold_path,
                            "valid_lab.npy",
                        )
                    ),
                ]
            )
        )
        Liz_X_val = np.load(os.path.join(params["data_path_liz"], "test_images.npy"))
        Liz_Y_val = add_3c_gt_fast(
            np.load(
                os.path.join(
                    params["data_path_liz"],
                    "test_labels.npy",
                )
            )
        )

    else:
        Liz_X_train = np.load(os.path.join(fold_path, "train_img.npy"))
        Liz_Y_train = add_3c_gt_fast(
            np.load(
                os.path.join(
                    fold_path,
                    "train_lab.npy",
                )
            )
        )
        Liz_X_val = np.load(os.path.join(fold_path, "valid_img.npy"))
        Liz_Y_val = add_3c_gt_fast(
            np.load(
                os.path.join(
                    fold_path,
                    "valid_lab.npy",
                )
            )
        )

    Mit_X_train = np.load(os.path.join(params["data_path_mit"], "train_full_img.npy"))
    Mit_X_val = np.load(os.path.join(params["data_path_mit"], "valid_full_img.npy"))

    Mit_Y_train = add_3c_gt_fast(
        np.load(os.path.join(params["hard_labels"], "train_full_lab.npy"))
    )
    Mit_Y_val = add_3c_gt_fast(
        np.load(os.path.join(params["hard_labels"], "valid_full_lab.npy"))
    )

    X_val = np.concatenate([Liz_X_val, Mit_X_val])
    Y_val = np.concatenate([Liz_Y_val, Mit_Y_val])

    labeled_dataset = SliceDataset(raw=Liz_X_train, labels=Liz_Y_train)
    validation_dataset = SliceDataset(raw=X_val, labels=Y_val)
    mit_dataset = SliceDataset(raw=Mit_X_train, labels=Mit_Y_train)
    # mit_val_dataset = SliceDataset(raw=Mit_X_val, labels=Mit_Y_val)
    labeled_dataloader = DataLoader(
        labeled_dataset,
        batch_size=params["batch_size"],
        prefetch_factor=2,
        sampler=get_weighted_sampler(labeled_dataset, classes=[0, 1, 2, 3, 4, 5, 6, 7]),
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    mit_labeled_dataloader = DataLoader(
        mit_dataset,
        batch_size=params["batch_size"],
        prefetch_factor=2,
        sampler=get_weighted_sampler(mit_dataset, classes=[0, 1, 2, 3, 4, 5, 6, 7]),
        num_workers=params["num_workers"],
        pin_memory=True,
    )

    dist_samp = DistributedSampler(
        validation_dataset,
        shuffle=True,
        drop_last=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        sampler=dist_samp,
        batch_size=params["validation_batch_size"],
        num_workers=params["num_workers"],
        pin_memory=True,
    )
    sz = (
        Liz_X_train.shape[0]
        if Liz_X_train.shape[0] < Mit_X_train.shape[0]
        else Mit_X_train.shape[0]
    )
    sz = int(sz / params["batch_size"])

    return (
        [labeled_dataloader, mit_labeled_dataloader],
        validation_dataloader,
        sz,
        dist_samp,
        CLASS_NAMES,
    )


def get_weighted_sampler(ds, classes=[0, 1, 2, 3, 4, 5, 6, 7]):
    count_list = []
    for gt in ds.labels:
        gt_classes = gt[..., 1].squeeze()
        tmp_list = []
        for c in classes:
            tmp_list.append(
                np.count_nonzero(gt_classes == c)
            )  # sum of individual classes for a sample
        count_list.append(np.stack(tmp_list))  #

    counts = np.stack(count_list)  # n_samples x classes
    sampling_weights = np.divide(
        counts,
        counts.sum(0)[np.newaxis, ...],
        where=counts.sum(0)[np.newaxis, ...] != 0,
    )  # n_samples x classes / 1 x classes = n_samples x classes
    sampling_weights = sampling_weights.sum(1)  # n_samples
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(sampling_weights),
        num_samples=len(sampling_weights),
        replacement=True,
    )
    return sampler
