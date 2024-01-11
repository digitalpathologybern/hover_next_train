import torch
import numpy as np
from tqdm.auto import tqdm


def batch_pseudolabel_ensemb(raw, models, nviews, aug, color_aug_fn):
    tmp_3c_view = []
    tmp_ct_view = []

    # disable TTA
    if nviews < 1:
        out_fast = []
        with torch.no_grad():
            for mod in models:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out_fast.append(mod(raw))
        out_fast = torch.stack(out_fast, axis=0).nanmean(0)
        tmp_3c_view.append(out_fast[:, 2:5].softmax(1))
        tmp_ct_view.append(out_fast[:, 5:].softmax(1))
    # enable TTA
    else:
        if nviews < len(models):
            nviews = len(models)

        for _ in range(nviews // len(models)):
            aug.interpolation = "bilinear"
            view_aug = aug.forward_transform(raw)
            aug.interpolation = "nearest"
            view_aug = color_aug_fn(view_aug)
            out_fast = []
            with torch.no_grad():
                for mod in models:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out_fast.append(aug.inverse_transform(mod(view_aug)))
            out_fast = torch.stack(out_fast, axis=0).nanmean(0)
            tmp_3c_view.append(out_fast[:, 2:5].softmax(1))
            tmp_ct_view.append(out_fast[:, 5:].softmax(1))
    return torch.stack(tmp_ct_view).nanmean(0), torch.stack(tmp_3c_view).nanmean(0)


def run_inference(dataloader, models, aug, color_aug_fn, tta=16, rank=0):
    pred_emb_list = []
    pred_class_list = []
    gt_list = []
    raw_list = []

    i = 0
    for raw, gt in tqdm(dataloader):
        raw = raw.to(rank).float()
        gt = gt.to(rank).float()
        raw = raw.permute(0, 3, 1, 2)  # BHWC -> BCHW
        gt = gt.permute(0, 3, 1, 2)  # BHW2 -> B2HW
        i += 1
        with torch.no_grad():
            pred_ct, pred_inst = batch_pseudolabel_ensemb(
                raw, models, tta, aug, color_aug_fn
            )
            pred_emb_list.append(pred_inst.squeeze().cpu().detach().numpy())
            pred_class_list.append(pred_ct.cpu().detach().numpy())
            gt_list.append(gt.permute(0, 2, 3, 1).cpu().detach().numpy())
            raw_list.append(raw.permute(0, 2, 3, 1).cpu().detach().numpy())

    pred_emb_list = np.concatenate(pred_emb_list)
    pred_class_list = np.concatenate(pred_class_list)
    gt_list = np.concatenate(gt_list)
    raw_list = np.concatenate(raw_list)
    return pred_emb_list, pred_class_list, gt_list, raw_list
