#!/usr/bin/env python3
"""
Training script for the Paired-Sampling Contrastive Framework.

Automatically detects available GPUs and uses DataParallel for multi-GPU training.

Usage:
    python tools/train.py --config configs/default.py
    python tools/train.py --config configs/default.py --gpus 0,1
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from mmengine.config import Config
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from src.dataset import build_datasets, build_dataloaders
from src.models import FeatHead
from src.loss_utils import BCEWithLogitsLoss_LS, FocalLoss, BinaryFocalLoss
from src.init_utils import set_seed, init_wandb
import src.training_procedure as trainers


def parse_args():
    parser = argparse.ArgumentParser(description="Train face attack detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.py",
        help="Path to config file",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPU IDs to use (e.g., '0' or '0,1'). If not specified, uses all available GPUs.",
    )
    return parser.parse_args()


def setup_device(gpus: str = None):
    """Setup CUDA devices and return device string."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu", 1

    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        num_gpus = len(gpus.split(","))
    else:
        num_gpus = torch.cuda.device_count()

    print(f"Using {num_gpus} GPU(s)")
    return "cuda", num_gpus


def train_main(config, num_gpus: int):
    """Main training function."""
    set_seed(config.dataset.get("val_seed", 42))

    # Build datasets
    train_ds, val_ds = build_datasets(config)
    train_loader, val_loader = build_dataloaders(config, train_ds, val_ds)

    # Build model
    model = FeatHead(
        backbone_name=config.model["model_name"],
        pretrained=config.model.get("pretrained", True),
        proj_dim=config.loss.get("supcon_proj_dim", 128),
    )

    # Multi-GPU support
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(config.device)

    # Optimizer
    optimizer_cls = getattr(torch.optim, config.optimizer.pop("type"))
    steps = len(train_loader) * config.trainer["epochs"]
    warmup_steps = int(0.05 * steps)
    optimizer = optimizer_cls(
        model.parameters(),
        lr=config.optimizer["lr"],
        weight_decay=config.optimizer["weight_decay"],
    )

    # Scheduler
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=steps,
        cycle_mult=1,
        max_lr=config.scheduler["max_lr"],
        min_lr=config.scheduler["min_lr"],
        warmup_steps=warmup_steps,
        gamma=config.scheduler["gamma"],
    )

    # Loss
    if config.loss["type"] == "BCE":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif config.loss["type"] == "binary_focal":
        criterion = BinaryFocalLoss(
            alpha=config.loss.get("alpha", 0.5),
            gamma=config.loss.get("gamma", 2.0),
        )
    elif config.loss["type"] == "focal":
        criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    else:
        criterion = BCEWithLogitsLoss_LS()

    # W&B init
    if config.wandb.get("use", False):
        init_wandb(config=config)

    # Trainer
    TrainerCls = getattr(trainers, config.trainer["type"])
    trainer = TrainerCls(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        schedulers=[scheduler],
        config=config,
        use_multihead=config.get("train_multihead", False),
        device=config.device,
        supcon_weight=config.loss.get("supcon_weight", 0.1),
        epochs=config.trainer["epochs"],
        output_name=config.wandb["experiment_name"],
        weights_save_folder=config.trainer["weights_save_folder"],
    )

    # Run training
    trainer.run(train_loader, val_loader)


if __name__ == "__main__":
    args = parse_args()
    device, num_gpus = setup_device(args.gpus)

    config = Config.fromfile(args.config)
    config.device = device

    train_main(config, num_gpus)
