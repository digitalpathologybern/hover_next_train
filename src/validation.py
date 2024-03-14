import numpy as np
from tqdm.auto import tqdm
from src.metrics import *
import torch
import torch.distributed as dist
from src.metrics import calc_MPQ, per_tile_metrics, get_output
from src.post_proc_utils import CLASS_NAMES
import pandas as pd
from skimage.segmentation import watershed
import cv2
from pannuke_metrics_master.variant import get_pannuke_pq


def make_instance_segmentation(prediction, fg_thresh=0.9, seed_thresh=0.9):
    fg = (1.0 - prediction[0, ...]) > fg_thresh
    ws_surface = 1.0 - prediction[1, ...]
    seeds = (prediction[1, ...] > seed_thresh).astype(np.uint8)
    _, markers = cv2.connectedComponents((seeds > 0).astype(np.uint8), connectivity=8)
    labelling = watershed(ws_surface, markers, mask=fg)
    return labelling, ws_surface


def make_ct(pred_class, instance_map):
    # device = pred_class.device
    pred_ct = torch.zeros_like(instance_map)
    pred_class_tmp = pred_class.softmax(1).squeeze(0)
    for instance in instance_map.unique():
        if instance == 0:
            continue
        ct = pred_class_tmp[:, instance_map == instance].sum(1)
        ct = ct.argmax()
        pred_ct[instance_map == instance] = ct

    return pred_ct


def make_prediction(pred_inst, pred_ct):
    pred_3c = pred_inst[:, 2:].softmax(1).squeeze().numpy()
    pred_inst, _ = make_instance_segmentation(pred_3c, fg_thresh=0.7, seed_thresh=0.3)
    pred_inst = pred_inst.astype(np.int32)  # torch.tensor(pred_inst).long()
    pred_class = pred_ct.softmax(1).squeeze()
    pred_ct = make_ct(pred_class, torch.from_numpy(pred_inst))
    return np.stack([pred_inst, pred_ct], axis=-1)


def validation(
    model,
    validation_dataloader,
    inst_lossfn,
    class_lossfn,
    device,
    step,
    world_size,
    nclasses=7,
    class_names=CLASS_NAMES,
    use_amp=True,
    metric="lizard",
):
    val_loss = []
    val_inst_loss = []
    val_ct_loss = []
    pred_list = []
    gt_list = []
    for raw, gt_b in tqdm(validation_dataloader):
        raw = raw.to(device).float()
        gt_b = gt_b.to(device)
        raw = raw.permute(0, 3, 1, 2)  # BHWC -> BCHW
        gt_b = gt_b.permute(0, 3, 1, 2)  # BHW2 -> B2HW
        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                out_ = model(raw)
            pred_inst = out_[:, :5]
            pred_class = out_[:, 5:]
            gt_inst = gt_b[:, 0]
            gt_ct = gt_b[:, 1]
            gt_3c = gt_b[:, 2]
            instance_loss = inst_lossfn(pred_inst, gt_inst, gt_3c)
            class_loss = class_lossfn(pred_class, gt_ct.long())
            loss = instance_loss + class_loss
            val_loss.append(loss)
            val_inst_loss.append(instance_loss.item())
            val_ct_loss.append(class_loss.item())
            out_ = out_.cpu().detach()
            gt_b = gt_b.cpu().detach()

            for out, gt in zip(out_, gt_b):
                out = out[None, :, :, :]
                gt = gt[None, :, :, :]
                pred_inst = out[:, :5]
                pred_class = out[:, 5:]
                pred = make_prediction(pred_inst, pred_class)
                pred_list.append(pred)
                gt_list.append(
                    gt[:, :2].squeeze().permute(1, 2, 0).cpu().detach().numpy()
                )

    val_new = torch.mean(torch.stack(val_loss)) / world_size
    dist.all_reduce(val_new, op=dist.ReduceOp.SUM)
    print("Step: ", step)
    if metric == "lizard":
        df, _ = calc_MPQ(pred_list, gt_list, nclasses)
        pq_p = torch.tensor(df["pq"][0]).to(device) / world_size
        mpq_p = torch.tensor(df["multi_pq+"][0]).to(device) / world_size
        dist.all_reduce(mpq_p, op=dist.ReduceOp.SUM)
        dist.all_reduce(pq_p, op=dist.ReduceOp.SUM)
        print("PQ:", pq_p)
        print("mPQ+", mpq_p)
        print("Validation loss: ", val_new)
        return mpq_p
    elif metric == "pannuke":
        pan_mpq, _, _, _ = get_pannuke_pq(gt_list, pred_list)
        pan_mpq = torch.tensor(pan_mpq).to(device) / world_size
        dist.all_reduce(pan_mpq, op=dist.ReduceOp.SUM)
        print("Pannuke mPQ:", pan_mpq)
        print("Validation loss: ", val_new)
        return pan_mpq
    else:
        metrics = per_tile_metrics(
            np.stack(gt_list, axis=0),
            np.stack(pred_list, axis=0),
            class_names[:nclasses],
        )
        _, metric_dict = get_output(metrics, None, class_names[:nclasses])
        f1 = (
            pd.DataFrame(metric_dict["count_metrics"])
            .set_index("class")
            .loc[class_names[:nclasses], "F1"]
            .values
        )
        hd = (
            pd.DataFrame(metric_dict["class_wise_seg_metrics"])
            .set_index("class")
            .loc[class_names[:nclasses], "seg_hausdorff_(TP)"]
            .values
        )
        f1_c = torch.from_numpy(f1).to(device) / world_size
        hd_c = torch.from_numpy(hd).to(device) / world_size
        dist.all_reduce(f1_c, op=dist.ReduceOp.SUM)
        dist.all_reduce(hd_c, op=dist.ReduceOp.SUM)
        print("F1:", f1_c)
        print("HD:", hd_c)
        print("Validation loss: ", val_new)
        return torch.mean(f1_c)
