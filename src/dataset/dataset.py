import os
import random
import csv
import cv2
import torch
from torch.utils.data import Dataset
from typing import Dict, Tuple, List, Optional

from .augs import live_aug_pipeline


def _load_bbox_csv(
    csv_path: str, root_dir: str = ""
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Build a lookup table: absolute_path → (x1, y1, x2, y2)

    Rows with missing coords OR any coordinate < 0 (e.g. -1 -1 -1 -1 from
    a failed detector) are skipped, so those images will fall back to
    full-frame processing downstream.
    """
    bbox_by_path: Dict[str, Tuple[float, float, float, float]] = {}

    if not csv_path:
        return bbox_by_path

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = (row.get("path") or "").strip()
            if not rel_path:
                continue

            try:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
            except (KeyError, TypeError, ValueError):
                continue

            if min(x1, y1, x2, y2) < 0:
                continue
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            full_path = os.path.join(root_dir, rel_path)
            bbox_by_path[full_path] = (x1, y1, x2, y2)

    return bbox_by_path


def _crop_with_margin(
    img, bbox: Tuple[float, float, float, float], margin: float
):
    """Crop *img* to *bbox* expanded by *margin* (e.g. 1.2 enlarges 20%).

    Falls back to returning *img* if bbox is invalid or crops outside bounds.
    """
    if bbox is None:
        return img

    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return img

    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    new_bw = bw * margin
    new_bh = bh * margin

    nx1 = max(0, int(cx - new_bw / 2.0))
    ny1 = max(0, int(cy - new_bh / 2.0))
    nx2 = min(w, int(cx + new_bw / 2.0))
    ny2 = min(h, int(cy + new_bh / 2.0))

    if nx2 <= nx1 or ny2 <= ny1:
        return img

    return img[ny1:ny2, nx1:nx2]


def read_protocol(protocol_path: str, root_dir: str = "") -> List[dict]:
    """
    Accepts either 2-column (path + label) or 1-column (path only) files.
    Returns list[dict] with keys: *filepath*, *code*, *class*.
    """
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            rel_path = parts[0]
            full_path = os.path.join(root_dir, rel_path)

            if len(parts) == 2:
                label_code = parts[1]
                binary_class = "live" if label_code == "0_0_0" else "attack"
            elif len(parts) == 1:
                label_code = None
                binary_class = None
            else:
                raise ValueError(f"Bad line in protocol file: {line}")

            entries.append(
                {"filepath": full_path, "code": label_code, "class": binary_class}
            )
    return entries


def preprocess_image(img, size: Tuple[int, int]):
    """BGR → RGB, then resize to `size` (H, W)."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size)
    return img_resized


class PairedAttackLiveDataset(Dataset):
    """
    CSV-driven (attack, live) pairs with optional face-box cropping.

    * Attack crops use the bbox in their own row.
    * Live crops fetch the bbox from the row whose `path` matches `paired_live_path`.

    A bbox with any coordinate < 0 (e.g. -1 -1 -1 -1) is treated as
    "no box → use full image".

    Each __getitem__ returns:
        ((attack_tensor, 1, attack_path), (live_tensor, 0, live_path))
    """

    def __init__(
        self,
        csv_file: str,
        model_input_size: Tuple[int, int],
        root_dir: str = "",
    ):
        self.model_input_size = model_input_size
        self.root_dir = root_dir

        self._rows: List[dict] = []
        self._bbox_by_path: Dict[str, Tuple[float, float, float, float]] = {}

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._rows.append(row)

                rel_path = (row.get("path") or "").strip()
                if not rel_path:
                    continue
                full_path = os.path.join(root_dir, rel_path)

                try:
                    x1 = float(row["x1"])
                    y1 = float(row["y1"])
                    x2 = float(row["x2"])
                    y2 = float(row["y2"])
                except (KeyError, TypeError, ValueError):
                    continue

                if min(x1, y1, x2, y2) < 0:
                    continue
                if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                    continue

                self._bbox_by_path[full_path] = (x1, y1, x2, y2)

        self._pairs: List[Tuple[str, str]] = []
        for row in self._rows:
            atk_rel = (row.get("path") or "").strip()
            live_rel = (row.get("paired_live_path") or "").strip()
            if not live_rel or live_rel.lower() in {"nan", "none"}:
                continue

            atk_path = os.path.join(root_dir, atk_rel)
            live_path = os.path.join(root_dir, live_rel)
            self._pairs.append((atk_path, live_path))

        if not self._pairs:
            raise ValueError(f"No paired samples found in {csv_file}!")

        self.sample_weights = [1.0] * len(self._pairs)

    def __len__(self):
        return len(self._pairs)

    def _to_tensor(self, img):
        img_resized = preprocess_image(img, self.model_input_size)
        return (
            torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        )

    def __getitem__(self, idx):
        atk_path, live_path = self._pairs[idx]

        # Attack branch
        atk_img = cv2.imread(atk_path)
        if atk_img is None:
            raise FileNotFoundError(f"Image not found: {atk_path}")

        atk_bbox = self._bbox_by_path.get(atk_path)
        atk_margin = random.uniform(1.1, 1.3)
        if atk_bbox is not None:
            atk_img = _crop_with_margin(atk_img, atk_bbox, atk_margin)

        atk_tensor = self._to_tensor(atk_img)

        # Live branch
        live_img = cv2.imread(live_path)
        if live_img is None:
            raise FileNotFoundError(f"Image not found: {live_path}")

        live_bbox = self._bbox_by_path.get(live_path)
        live_margin = random.uniform(1.1, 1.3)
        if live_bbox is not None:
            live_img = _crop_with_margin(live_img, live_bbox, live_margin)

        live_img = live_aug_pipeline(image=live_img)["image"]
        live_tensor = self._to_tensor(live_img)

        return (atk_tensor, 1, atk_path), (live_tensor, 0, live_path)


class ValDataset(Dataset):
    """Standard validation set, optionally bbox-aware (margin = 1.2)."""

    LABEL_MAPPING = {"live": 0, "attack": 1}

    def __init__(
        self,
        protocol_file: str,
        model_input_size: Tuple[int, int],
        root_dir: str = "",
        bbox_csv: str = None,
    ):
        self.entries = read_protocol(protocol_file, root_dir=root_dir)
        self.model_input_size = model_input_size
        self.bbox_by_path = _load_bbox_csv(bbox_csv, root_dir) if bbox_csv else {}
        self.margin = 1.2

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        row = self.entries[idx]
        path = row["filepath"]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")

        bbox = self.bbox_by_path.get(path)
        if bbox is not None:
            img = _crop_with_margin(img, bbox, self.margin)
        img_resized = preprocess_image(img, self.model_input_size)
        img_tensor = (
            torch.tensor(img_resized.transpose(2, 0, 1), dtype=torch.float32) / 255.0
        )

        cls = row["class"]
        if cls is None:
            raise ValueError(f"No class for: {path}")
        label = self.LABEL_MAPPING[cls]
        return img_tensor, label, path
