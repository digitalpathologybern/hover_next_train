import torch
import torch.nn.functional as F
from src.data_utils import center_crop, parallel_cpvs


def save_model(step, model, optimizer, loss, best_loss, filename):
    torch.save(
        {
            "step": step,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "best_loss": best_loss,
        },
        filename,
    )


def supervised_train_step(
    model,
    raw,
    gt,
    fast_aug,
    color_aug_fn,
    inst_lossfn,
    class_lossfn,
    device,
    params,
):
    raw = raw.to(device).float().permute(0, 3, 1, 2)
    gt = gt.to(device).float().permute(0, 3, 1, 2)
    if gt.shape[-1] > 2:
        cpv_3c_model = True
    else:
        cpv_3c_model = False

    img_saug, gt_saug = fast_aug.forward_transform(raw, gt)
    gt_inst = gt_saug[:, 0]
    gt_ct = gt_saug[:, 1]
    if cpv_3c_model:
        gt_3c = gt_saug[:, 2]
    img_caug = torch.clamp(color_aug_fn(img_saug), 0, 1)
    # img_caug = normalization(img_caug)

    with torch.autocast(
        device_type="cuda", dtype=torch.float16, enabled=params["use_amp"]
    ):
        out_fast = model(img_caug)
        _, _, H, W = out_fast.shape

        gt_inst = center_crop(gt_inst, H, W)
        gt_ct = center_crop(gt_ct, H, W)

        pred_inst = out_fast[:, : params["inst_channels"]]
        pred_class = out_fast[:, params["inst_channels"] :]

        # gt_3c = torch.cat(gt_3c_list, axis=0)
        gt_3c = center_crop(gt_3c, H, W)
        instance_loss = inst_lossfn(pred_inst, gt_inst, gt_3c)
        class_loss = class_lossfn(pred_class, gt_ct.long())
        loss = (
            params["loss_lambda"] * instance_loss
            + (1 - params["loss_lambda"]) * class_loss
        )
    print("inst_loss: ", instance_loss.item(), "class_loss: ", class_loss.item())
    return loss


class InstanceLoss(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.label_smoothing = params["label_smoothing"]
        self.dataset = params["dataset"]
        self.loss_fn = (
            self.inst_loss_fn_lizard
            if self.dataset == "lizard"
            else self.inst_loss_fn_pannuke
        )

    def inst_loss_fn_lizard(self, input, gt_inst, gt_3c):
        gt_cpv = parallel_cpvs(gt_inst.to("cpu")).to(gt_inst.device)
        loss_cpv = F.mse_loss(input=input[:, :2], target=gt_cpv)
        loss_3c = F.cross_entropy(
            input=input[:, 2:],
            target=gt_3c.long(),
            weight=torch.tensor([1, 1, 2]).type_as(input).to(input.device),
        )
        return loss_cpv + loss_3c

    def inst_loss_fn_pannuke(self, input, gt_inst, gt_3c):
        gt_cpv = parallel_cpvs(gt_inst.to("cpu")).to(gt_inst.device)
        loss_cpv = F.smooth_l1_loss(input=input[:, :2], target=gt_cpv)
        loss_3c = F.cross_entropy(
            input=input[:, 2:],
            target=gt_3c.long(),
            weight=torch.tensor([1, 1, 2]).type_as(input).to(input.device),
            label_smoothing=self.label_smoothing,
        )
        return loss_cpv + loss_3c

    def forward(self, input, gt_inst, gt_3c):
        return self.loss_fn(input, gt_inst, gt_3c)
