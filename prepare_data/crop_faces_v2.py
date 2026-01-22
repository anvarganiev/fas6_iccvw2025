#!/usr/bin/env python3
# crop_faces_to_df.py
"""
Crop faces *and* save all bounding boxes into a single CSV.

CSV schema
----------
path,x1,y1,x2,y2      # coordinates refer to the *original* image

* -1 -1 -1 -1  → “no face detected”

Example
-------
python crop_faces_to_df.py \
    --src  /data/raw_images \
    --dst  /data/crops224 \
    --out-size 224 \
    --device cuda
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision.transforms import ToPILImage
from tqdm import tqdm


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser("Crop faces + export bounding-boxes CSV.")
    p.add_argument("--src", required=True, help="Source root with images")
    p.add_argument("--dst", required=True, help="Destination root for crops")
    p.add_argument("--out-size", type=int, default=224,
                   help="Square output size (0 = keep original)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ext", nargs="+", default=["jpg", "jpeg", "png", "bmp"])
    p.add_argument("--csv-name", default="bboxes.csv",
                   help="Filename for the bbox CSV (inside dst)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
def clip_box(box: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    return x1, y1, x2, y2


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    src_root = Path(args.src).expanduser().resolve()
    dst_root = Path(args.dst).expanduser().resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    mtcnn   = MTCNN(keep_all=False, device=device)
    to_pil  = ToPILImage()

    img_files: List[Path] = [p for p in src_root.rglob("*")
                             if p.suffix[1:].lower() in args.ext]

    records: List[Dict[str, int | str]] = []

    for img_path in tqdm(img_files, desc="Cropping"):
        # read
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # detect
        try:
            boxes, _ = mtcnn.detect(img)          # ndarray (n,4) or None
        except RuntimeError:
            boxes = None

        if boxes is not None:
            # choose largest box if multiple
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            box   = boxes[areas.argmax()]
            x1, y1, x2, y2 = clip_box(box, img.width, img.height)
            crop = img.crop((x1, y1, x2, y2))
        else:
            crop = img
            x1 = y1 = x2 = y2 = -1  # sentinel “no face”

        if args.out_size > 0:
            crop = crop.resize((args.out_size, args.out_size), Image.BILINEAR)

        # save crop
        out_img = dst_root / img_path.relative_to(src_root)
        out_img = out_img.with_suffix(".jpg")
        out_img.parent.mkdir(parents=True, exist_ok=True)
        crop.save(out_img, quality=95)

        # record bbox
        records.append({
            "path": str(img_path.relative_to(src_root)),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    # write DataFrame
    df = pd.DataFrame.from_records(records)
    csv_path = dst_root / args.csv_name
    df.to_csv(csv_path, index=False)
    print(f"✓ Finished. Crops saved under {dst_root}\n"
          f"  Bounding boxes ➜ {csv_path}")


if __name__ == "__main__":
    main()