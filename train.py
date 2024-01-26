import os
import random
import sys
import argparse
import toml
import numpy as np
import torch
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from src.color_conversion import color_augmentations  # , get_normalize
from src.multi_head_unet import get_model, load_checkpoint, freeze_enc, unfreeze_enc
from src.spatial_augmenter import SpatialAugmenter
from src.train_utils import (
    supervised_train_step,
    save_model,
    InstanceLoss,
)
from src.validation import validation
from src.data_utils import get_data
from src.focal_loss import FocalLoss


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["OMP_NUM_THREADS"] = "16"
random.seed(42)

dist.init_process_group("nccl")
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)


def newest(path):
    files = [
        file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))
    ]
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)


def supervised_training(params):
    # initialize environment
    torch.set_num_threads(params["num_workers"])
    validation_loss = []
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    print(rank, flush=True)

    # build model and retrieve weights
    model = get_model(
        enc=params["encoder"],
        out_channels_cls=params["out_channels_cls"],
        out_channels_inst=params["inst_channels"],
        pretrained=params["pretrained"],
    ).to(rank)

    if "checkpoint_path" in params.keys() and params["checkpoint_path"]:
        model, step, best_loss = load_checkpoint(
            model, params["checkpoint_path"], rank=0
        )
        params["step"] = step
        validation_loss.append(best_loss)
    model.train()

    ddp_model = DDP(model, find_unused_parameters=True)

    # setup training
    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
        eps=1e-4,
    )

    scaler = GradScaler()

    if "checkpoint_path" in params.keys() and params["checkpoint_path"]:
        optimizer.load_state_dict(
            torch.load(params["checkpoint_path"], map_location="cpu")[
                "optimizer_state_dict"
            ]
        )
        print("Load optimizer state dict")

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["training_steps"], eta_min=params["min_learning_rate"]
    )
    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1.0 / 1000, 1.0)
    ce_loss_fn = FocalLoss(alpha=None, gamma=params["fl_gamma"], reduction="mean").to(
        rank
    )
    inst_loss_fn = InstanceLoss(params)

    # setup augmentation functions
    color_aug_fn = color_augmentations(True, s=params["color_scale"], rank=rank)
    fast_aug = SpatialAugmenter(params["aug_params_fast"], random_seed=params["seed"])
    # normalization = get_normalize(use_norm=params["dataset"] == "pannuke")

    # load data
    train_dataloaders, validation_dataloader, sz, dist_samp, class_names = get_data(
        params
    )
    if "step" in params.keys() and params["step"] != None:
        step = params["step"]
    else:
        step = -1
    print("start step", step)
    ep_cnt = 0

    # for debugging
    na_steps = []

    # warmup
    freeze_enc(ddp_model.module)
    # train loop
    while step < params["training_steps"]:
        train_loaders = [iter(x) for x in train_dataloaders]

        for _ in range(sz):
            # stop warmup
            if step == params["warmup_steps"]:
                print("Warmup steps reached, unfreezing encoder weights...")
                unfreeze_enc(ddp_model.module)
            # sample from the available datasets:
            raw, gt = next(train_loaders[random.randint(0, len(train_loaders) - 1)])
            step += 1
            for param in ddp_model.parameters():
                param.grad = None
            loss = supervised_train_step(
                ddp_model,
                raw,
                gt,
                fast_aug,
                color_aug_fn,
                inst_loss_fn,
                ce_loss_fn,
                rank,
                params,
            )
            if not torch.isfinite(loss):
                na_steps.append(1)
            if len(na_steps) > 10:
                raise ValueError(
                    "Too many NaN steps, something is wrong with the model training"
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            # if step < params["warmup_steps"]:
            # warmup_scheduler.step()
            # else:
            lr_scheduler.step()

            if step % (params["validation_step"] // world_size) == 0:
                dist_samp.set_epoch(ep_cnt)
                ddp_model.eval()
                val_new = validation(
                    ddp_model,
                    validation_dataloader,
                    inst_loss_fn,
                    ce_loss_fn,
                    rank,
                    step,
                    world_size,
                    nclasses=len(class_names),
                    class_names=class_names,
                    use_amp=True,
                    metric=params["optim_metric"],
                )
                # CAREFUL: If you want to minimize a metric instead, you need to edit the checkpoint loading
                if rank == 0:
                    val_new = val_new.cpu().numpy()
                    validation_loss.append(val_new)
                    if val_new >= np.max(validation_loss):
                        print("Save best model")
                        save_model(
                            step,
                            ddp_model,
                            optimizer,
                            loss,
                            val_new,
                            os.path.join(log_dir, "best_model"),
                        )
                ep_cnt += 1
                ddp_model.train()
                sys.stdout.flush()
                dist.barrier()
            if step % (params["checkpoint_step"] // world_size) == 0:
                if rank == 0:
                    save_model(
                        step,
                        ddp_model,
                        optimizer,
                        loss,
                        np.max(validation_loss),
                        os.path.join(log_dir, "checkpoint_step_" + str(step)),
                    )
                sys.stdout.flush()
                dist.barrier()


def main(params):
    supervised_training(params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id",
        type=int,
        default=1,
        help="train id, used to resume training, set to higher than 1 to resume training",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sample_configs/train_lizard.toml",
        help="config file with hyperparameters, check sample_configs folder for examples",
    )
    args = parser.parse_args()

    print(torch.cuda.device_count(), " cuda devices")

    params = toml.load(args.config)
    params["experiment"] = params["experiment"] + "_" + str(params["fold"])

    if int(args.id) > 1:
        params["checkpoint_path"] = newest(params["experiment"] + "/train/")

    # creating experiment directory and storing parameters
    log_dir = os.path.join(params["experiment"], "train")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(params["experiment"], "params.toml"), "w") as f:
        toml.dump(params, f)

    main(params)
