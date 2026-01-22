#!/usr/bin/env python3
"""

Usage example
-------------
```bash

```
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TRAIN_PREFIX = "Data-train/"
VAL_PREFIX   = "Data-val/"
TEST_PREFIX  = "Data-test/"


def parse_args() -> argparse.Namespace:  # noqa: D401
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge per‑image bounding‑boxes into protocol CSVs (train/val/test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Primary CSV inputs
    parser.add_argument("--train-df", type=Path, required=True, help="Train protocol CSV.")
    parser.add_argument("--val-df",   type=Path, required=True, help="Validation protocol CSV.")
    parser.add_argument("--test-df",  type=Path, required=True, help="Test protocol CSV.")

    # Bounding‑box CSV inputs
    parser.add_argument("--train-bboxes", type=Path, required=True, help="Train bounding‑boxes CSV.")
    parser.add_argument("--val-bboxes",   type=Path, required=True, help="Validation bounding‑boxes CSV.")
    parser.add_argument("--test-bboxes",  type=Path, required=True, help="Test bounding‑boxes CSV.")

    # Output CSV paths (train mandatory, val/test optional)
    parser.add_argument(
        "--output-train-csv",
        type=Path,
        required=True,
        help="Destination CSV for the merged train dataframe.",
    )
    parser.add_argument("--output-val-csv",  type=Path, default=None, help="If set, write merged val CSV here.")
    parser.add_argument("--output-test-csv", type=Path, default=None, help="If set, write merged test CSV here.")

    return parser.parse_args()


def prepend_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return a copy of *df* with *prefix* prepended to the ``path`` column."""
    df = df.copy()
    df["path"] = prefix + df["path"].astype(str)
    return df


def merge_bbox(protocol_df: pd.DataFrame, bbox_df: pd.DataFrame) -> pd.DataFrame:
    """Left‑join *protocol_df* with *bbox_df* on the ``path`` column."""
    return protocol_df.merge(bbox_df, on="path", how="left")


def main() -> None:
    args = parse_args()

    train_df  = pd.read_csv(args.train_df)
    val_df    = pd.read_csv(args.val_df, names=["path", "label_original"], header=None, sep=" ")
    test_df   = pd.read_csv(args.test_df, names=["path", "label_original"], header=None, sep=" ")

    train_bbx = pd.read_csv(args.train_bboxes)
    val_bbx   = pd.read_csv(args.val_bboxes)
    test_bbx  = pd.read_csv(args.test_bboxes)

    train_bbx = prepend_prefix(train_bbx, TRAIN_PREFIX)
    val_bbx   = prepend_prefix(val_bbx, VAL_PREFIX)
    test_bbx  = prepend_prefix(test_bbx, TEST_PREFIX)

    train_merged = merge_bbox(train_df, train_bbx)
    val_merged   = merge_bbox(val_df,   val_bbx)
    test_merged  = merge_bbox(test_df,  test_bbx)


    args.output_train_csv.parent.mkdir(parents=True, exist_ok=True)
    train_merged.to_csv(args.output_train_csv, index=False)
    print(f"Saved TRAIN dataframe to: {args.output_train_csv}")

    if args.output_val_csv is not None:
        args.output_val_csv.parent.mkdir(parents=True, exist_ok=True)
        val_merged.to_csv(args.output_val_csv, index=False)
        print(f"Saved VAL dataframe to:   {args.output_val_csv}")

    if args.output_test_csv is not None:
        args.output_test_csv.parent.mkdir(parents=True, exist_ok=True)
        test_merged.to_csv(args.output_test_csv, index=False)
        print(f"Saved TEST dataframe to:  {args.output_test_csv}")


if __name__ == "__main__":
    main()
