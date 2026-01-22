from __future__ import annotations

import os
from typing import List

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from mmengine.config import Config

from .dataset import ValDataset, PairedAttackLiveDataset

__all__ = ["build_datasets", "build_dataloaders"]


def my_collate(batch):
    """Smart collate that handles paired vs single-sample entries."""
    # Paired set → flatten
    if isinstance(batch[0][0], tuple):
        flat = [item for pair in batch for item in pair]
        if len(flat[0]) == 3:
            imgs, labels, paths = zip(*flat)
            return torch.stack(imgs), torch.tensor(labels), list(paths)
        imgs, labels = zip(*flat)
        return torch.stack(imgs), torch.tensor(labels)

    # Single-sample sets
    if len(batch[0]) == 3:
        imgs, labels, paths = zip(*batch)
        return torch.stack(imgs), torch.tensor(labels), list(paths)

    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels)


def _resolve_path(path: str, root_dir: str) -> str:
    """Make *path* absolute if it's relative (rooted at *root_dir*)."""
    if path and not os.path.isabs(path):
        return os.path.join(root_dir, path)
    return path


def build_datasets(config: Config):
    """Create lists of training and validation datasets."""
    root_dir: str = config.paths.get("root_dir", "")
    model_size = config.model_input_size

    # Train (paired CSV)
    train_datasets: List[PairedAttackLiveDataset] = []
    csvs = config.paths.get("train_csv")
    if csvs is None:
        raise ValueError("`train_csv` missing in config.paths")
    if not isinstance(csvs, (list, tuple)):
        csvs = [csvs]

    for csv_file in csvs:
        ds = PairedAttackLiveDataset(
            csv_file=_resolve_path(csv_file, root_dir),
            model_input_size=model_size,
            root_dir=root_dir,
        )
        train_datasets.append(ds)
        print(f"Train CSV: {os.path.basename(csv_file)} → {len(ds)} samples")

    # Validation (protocol)
    val_datasets: List[ValDataset] = []
    protocols = config.paths.get("val_protocol")
    if protocols is None:
        raise ValueError("You must provide `val_protocol` in config.paths")
    if not isinstance(protocols, (list, tuple)):
        protocols = [protocols]

    val_bbox_csv = _resolve_path(config.paths.get("val_bbox_csv"), root_dir)

    for proto in protocols:
        ds = ValDataset(
            protocol_file=_resolve_path(proto, root_dir),
            model_input_size=model_size,
            root_dir=root_dir,
            bbox_csv=val_bbox_csv,
        )
        val_datasets.append(ds)
        print(f"Val: {os.path.basename(proto)} → {len(ds)} samples")

    return train_datasets, val_datasets


def build_dataloaders(
    config: Config,
    train_datasets: List[PairedAttackLiveDataset],
    val_datasets: List[ValDataset],
):
    """Return PyTorch DataLoaders for train / val."""
    train_ds = ConcatDataset(train_datasets)

    all_weights: List[float] = []
    for ds in train_datasets:
        all_weights.extend(getattr(ds, "sample_weights", [1.0] * len(ds)))

    epoch_fraction = config.dataset.get("epoch_fraction", 0.5)
    num_samples = int(epoch_fraction * len(all_weights))

    sampler = WeightedRandomSampler(
        all_weights, num_samples=num_samples, replacement=False
    )

    train_loader = DataLoader(
        train_ds,
        sampler=sampler,
        collate_fn=my_collate,
        **{k: v for k, v in config.train_dataloader.items() if k != "shuffle"},
    )

    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        collate_fn=my_collate,
        **config.val_dataloader,
    )

    return train_loader, val_loader
