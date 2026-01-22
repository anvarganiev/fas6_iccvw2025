#!/usr/bin/env python
# extract_face_embeddings.py
"""
Usage
-----
pip install facenet-pytorch pillow numpy tqdm
python extract_face_embeddings.py \
        --src  data/raw_images \
        --dst  data/embeddings \
        --batch-size 32 \
        --device cuda      # or cpu
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract 512-D face embeddings")
    p.add_argument("--src", required=True, help="Root folder with images")
    p.add_argument("--dst", required=True, help="Destination root for .npy files")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for embedding network")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="Device: cuda / cpu")
    p.add_argument("--ext", nargs="+", default=["jpg", "jpeg", "png", "bmp", "tif"],
                   help="Image file extensions")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Helper to load and detect                                                   #
# --------------------------------------------------------------------------- #
def load_images(paths: List[Path]) -> List[Image.Image]:
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            imgs.append(None)
    return imgs


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    # -------- models --------
    device = torch.device(args.device)
    mtcnn = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # -------- files list --------
    img_files = [p for p in src_root.rglob("*")
                 if p.suffix.lower()[1:] in args.ext]

    batch, batch_paths = [], []
    for img_path in tqdm(img_files, desc="Embedding"):
        img = Image.open(img_path).convert("RGB")
        face = mtcnn(img)                 # tensor C×H×W or None
        if face is None:
            continue                      # skip images with no face

        batch.append(face)
        batch_paths.append(img_path)

        # whenever batch full → embed
        if len(batch) == args.batch_size:
            process_batch(batch, batch_paths, resnet, dst_root, src_root, device)
            batch, batch_paths = [], []

    # last partial batch
    if batch:
        process_batch(batch, batch_paths, resnet, dst_root, src_root, device)

    print(f"Done. Embeddings saved in {dst_root}")


# --------------------------------------------------------------------------- #
def process_batch(
    tensor_list: List[torch.Tensor],
    paths: List[Path],
    net: InceptionResnetV1,
    dst_root: Path,
    src_root: Path,
    device: torch.device,
):
    stacked = torch.stack(tensor_list).to(device)  # B×3×160×160
    with torch.no_grad():
        emb = net(stacked)                         # B×512
        emb = F.normalize(emb, p=2, dim=1)         # L2-norm

    emb_np = emb.cpu().numpy()
    for e, p in zip(emb_np, paths):
        out_path = dst_root / p.relative_to(src_root)
        out_path = out_path.with_suffix(".npy")
        ensure_parent(out_path)
        np.save(out_path, e)


if __name__ == "__main__":
    main()