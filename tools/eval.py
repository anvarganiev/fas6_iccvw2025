#!/usr/bin/env python3
"""
Evaluation script for the Paired-Sampling Contrastive Framework.

Usage:
    python tools/eval.py --checkpoint weights/pt/best.pth --protocol dataset/Protocol-val-test.txt
    python tools/eval.py --checkpoint weights/pt/best.pth --protocol dataset/Protocol-test.txt --output submission.txt
"""
import os
import sys
import csv
import zipfile
import argparse
from typing import Dict, Tuple, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FeatHead(nn.Module):
    """Model with feature head for contrastive learning."""

    def __init__(self, backbone_name: str, pretrained: bool = True, proj_dim: int = 128):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        in_f = self.backbone.num_features
        self.cls = nn.Linear(in_f, 1)
        self.proj = nn.Sequential(
            nn.Linear(in_f, in_f // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_f // 2, proj_dim),
        )

    def forward(self, x, return_feats: bool = False):
        f = self.backbone(x)
        logits = self.cls(f)
        if return_feats:
            feats = F.normalize(self.proj(f), dim=1)
            return logits, feats
        return logits


def _load_bbox_csv(csv_path: str, root_dir: str = "") -> Dict[str, Tuple[float, float, float, float]]:
    """Load bounding box CSV into a lookup table."""
    bbox_by_path = {}

    if not csv_path or not os.path.exists(csv_path):
        return bbox_by_path

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = (row.get("path") or "").strip()
            if not rel_path:
                continue

            try:
                x1, y1 = float(row["x1"]), float(row["y1"])
                x2, y2 = float(row["x2"]), float(row["y2"])
            except (KeyError, TypeError, ValueError):
                continue

            if min(x1, y1, x2, y2) < 0 or (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            full_path = os.path.join(root_dir, rel_path)
            bbox_by_path[full_path] = (x1, y1, x2, y2)

    return bbox_by_path


def _crop_with_margin(img, bbox: Tuple[float, float, float, float], margin: float):
    """Crop image to bbox expanded by margin."""
    if bbox is None:
        return img

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return img

    cx, cy = x1 + bw / 2.0, y1 + bh / 2.0
    new_bw, new_bh = bw * margin, bh * margin

    nx1 = max(0, int(cx - new_bw / 2.0))
    ny1 = max(0, int(cy - new_bh / 2.0))
    nx2 = min(w, int(cx + new_bw / 2.0))
    ny2 = min(h, int(cy + new_bh / 2.0))

    if nx2 <= nx1 or ny2 <= ny1:
        return img

    return img[ny1:ny2, nx1:nx2]


def read_protocol(protocol_path: str, root_dir: str = "") -> List[str]:
    """Read protocol file and return list of file paths."""
    files = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                files.append(os.path.join(root_dir, parts[0]))
    return files


class InferenceDataset(Dataset):
    """Dataset for inference without labels."""

    def __init__(
        self,
        files: List[str],
        model_input_size: Tuple[int, int],
        bbox_by_path: Dict = None,
        margin: float = 1.2,
    ):
        self.files = files
        self.model_input_size = model_input_size
        self.bbox_by_path = bbox_by_path or {}
        self.margin = margin

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")

        bbox = self.bbox_by_path.get(path)
        if bbox is not None:
            img = _crop_with_margin(img, bbox, self.margin)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.model_input_size)
        img_tensor = torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        return img_tensor, path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate face attack detection model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--protocol", type=str, required=True, help="Path to protocol file")
    parser.add_argument("--root-dir", type=str, default="", help="Root directory for images")
    parser.add_argument("--bbox-csv", type=str, default=None, help="Path to bounding box CSV")
    parser.add_argument("--output", type=str, default="submission.txt", help="Output submission file")
    parser.add_argument("--backbone", type=str, default="convnextv2_tiny", help="Backbone model name")
    parser.add_argument("--proj-dim", type=int, default=128, help="Projection dimension")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loader workers")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--no-zip", action="store_true", help="Do not create zip file")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = FeatHead(
        backbone_name=args.backbone,
        proj_dim=args.proj_dim,
        pretrained=False,
    ).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    if next(iter(state)).startswith("module."):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load data
    files = read_protocol(args.protocol, args.root_dir)
    bbox_by_path = _load_bbox_csv(args.bbox_csv, args.root_dir) if args.bbox_csv else {}

    dataset = InferenceDataset(
        files=files,
        model_input_size=(args.image_size, args.image_size),
        bbox_by_path=bbox_by_path,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Evaluating {len(dataset)} images...")

    # Inference
    all_scores = []
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for images, paths in dataloader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = sigmoid(logits)
            all_scores.extend(probs.cpu().tolist())

    # Read original protocol lines for output
    with open(args.protocol, "r") as f:
        protocol_lines = [ln.strip().split()[0] for ln in f if ln.strip()]

    # Write submission
    sub_lines = [f"{path} {score:.5f}" for path, score in zip(protocol_lines, all_scores)]

    with open(args.output, "w") as f:
        f.write("\n".join(sub_lines))

    print(f"Submission saved: {args.output}")

    # Create zip
    if not args.no_zip:
        zip_path = args.output.replace(".txt", ".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.output, arcname=os.path.basename(args.output))
        print(f"Zip created: {zip_path} ({os.path.getsize(zip_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
