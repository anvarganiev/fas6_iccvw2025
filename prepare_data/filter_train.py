#!/usr/bin/env python3
"""
Filter attack samples by cosine similarity to live samples.

This script is a command‑line wrapper around the original notebook logic shared in our conversation. It performs the following steps:

1. Load a cleaned training CSV with image relative paths and labels.
2. Resolve absolute image paths using ``--data-dir``.
3. Load per‑image embeddings stored as ``.npy`` files under ``--embeddings-dir``.
4. Compute cosine similarity between attack and live embeddings.
5. Keep attacks whose highest similarity to any live sample is at least ``--sim-thr``.
6. Save the resulting dataframe (all live samples + filtered attacks) to ``--output-csv``.

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


def parse_args() -> argparse.Namespace:  # noqa: D401
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter attack images by their cosine similarity to live images "
                    "using pre‑computed embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing the images referred to by the CSV's `path` column.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="Directory that mirrors the image directory structure, containing .npy embeddings.",
    )
    parser.add_argument(
        "--clean-df-path",
        type=Path,
        required=True,
        help="CSV file produced after cleaning label noise (e.g. train_no_misslabel.csv).",
    )
    parser.add_argument(
        "--sim-thr",
        type=float,
        default=0.84,
        help="Cosine‑similarity threshold used to keep attack samples.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("train_matched.csv"),
        help="Destination CSV file with the resulting (live + filtered attack) samples.",
    )
    parser.add_argument(
        "--progress-desc",
        type=str,
        default="Loading embeddings",
        help="Description shown next to the tqdm progress bar.",
    )
    return parser.parse_args()


def load_emb(img_path: Path, img_root: Path, embeddings_dir: Path) -> Optional[np.ndarray]:
    """Load a single embedding if it exists; return *None* otherwise."""
    # Reproduce original logic: images might be nested; keep only filename (stem)
    emb_path = embeddings_dir / img_path.relative_to(img_root).stem
    emb_path = emb_path.with_suffix(".npy")
    return np.load(emb_path) if emb_path.exists() else None


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------------
    # 1. Read metadata CSV and construct absolute image paths
    # ---------------------------------------------------------------------
    df = pd.read_csv(args.clean_df_path)
    img_root: Path = args.data_dir
    df["abs_path"] = df["path"].apply(lambda p: img_root / p)

    # ---------------------------------------------------------------------
    # 2. Load embeddings (only those that actually exist)
    # ---------------------------------------------------------------------
    emb_list: List[np.ndarray] = []
    idx_list: List[int] = []
    for idx, path in tqdm(
        enumerate(df["abs_path"]),
        total=len(df),
        desc=args.progress_desc,
    ):
        emb = load_emb(path, img_root, args.embeddings_dir)
        if emb is not None:
            emb_list.append(emb)
            idx_list.append(idx)

    if not emb_list:
        raise RuntimeError("No embeddings were loaded. Check --embeddings-dir and image paths.")

    emb_matrix = np.stack(emb_list)
    df = df.loc[idx_list].reset_index(drop=True)

    # ---------------------------------------------------------------------
    # 3. Split into live / attack subsets based on original label
    # ---------------------------------------------------------------------
    lives_mask = df["label_original"] == "0_0_0"
    attacks_mask = ~lives_mask

    live_embs = emb_matrix[lives_mask.values]
    attack_embs = emb_matrix[attacks_mask.values]

    live_rows = df[lives_mask].reset_index(drop=True)
    attack_rows = df[attacks_mask].reset_index(drop=True)

    # ---------------------------------------------------------------------
    # 4. Similarity search: for each attack, find its most similar live sample
    # ---------------------------------------------------------------------
    sim_mat = cosine_similarity(attack_embs, live_embs)
    best_live_idx = sim_mat.argmax(axis=1)
    best_live_sim = sim_mat[np.arange(len(sim_mat)), best_live_idx]
    best_live_paths = live_rows.loc[best_live_idx, "path"].values

    attack_rows["max_live_sim"] = best_live_sim
    attack_rows["paired_live_path"] = best_live_paths

    # ---------------------------------------------------------------------
    # 5. Filter attacks by similarity threshold and merge with all lives
    # ---------------------------------------------------------------------
    filtered_attacks = attack_rows[attack_rows["max_live_sim"] >= args.sim_thr]
    print(
        f"kept {len(filtered_attacks)} / {len(attack_rows)} attacks "
        f"(sim \u2265 {args.sim_thr})"
    )

    df_result = pd.concat([live_rows, filtered_attacks], ignore_index=True)

    # ---------------------------------------------------------------------
    # 6. Persist results
    # ---------------------------------------------------------------------
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(args.output_csv, index=False)
    print(f"Saved result to: {args.output_csv}")


if __name__ == "__main__":
    main()
