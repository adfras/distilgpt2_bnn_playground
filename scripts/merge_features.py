#!/usr/bin/env python
"""Merge multiple token feature CSVs and annotate with prompt metadata."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd


def load_meta(csv_path: Path, meta_path: Path | None) -> dict:
    if meta_path is None:
        meta_path = csv_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found for {csv_path}: {meta_path}")
    with meta_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge token feature CSVs with prompt metadata")
    parser.add_argument(
        "inputs",
        type=Path,
        nargs='+',
        help="Input feature CSV paths",
    )
    parser.add_argument(
        "--meta-paths",
        type=Path,
        nargs='*',
        default=None,
        help="Optional explicit metadata JSON paths matching the inputs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/token_features_merged.csv"),
        help="Output merged CSV",
    )
    parser.add_argument(
        "--meta-output",
        type=Path,
        default=Path("data/token_features_merged.meta.json"),
        help="Output metadata JSON summarising the merge",
    )
    args = parser.parse_args()

    meta_paths: List[Path | None]
    if args.meta_paths is None:
        meta_paths = [None] * len(args.inputs)
    else:
        if len(args.meta_paths) != len(args.inputs):
            raise ValueError("Number of meta paths must match number of inputs")
        meta_paths = list(args.meta_paths)

    frames = []
    prompt_entries = []
    for csv_path, meta_override in zip(args.inputs, meta_paths):
        meta = load_meta(csv_path, meta_override)
        df = pd.read_csv(csv_path)
        df["prompt_hash"] = meta.get("prompt_hash")
        df["prompt"] = meta.get("prompt")
        df["source_csv"] = str(csv_path)
        frames.append(df)
        prompt_entries.append(
            {
                "source_csv": str(csv_path),
                "prompt_hash": meta.get("prompt_hash"),
                "prompt": meta.get("prompt"),
                "seed": meta.get("seed"),
                "max_new_tokens": meta.get("max_new_tokens"),
                "temperature": meta.get("temperature"),
                "top_p": meta.get("top_p"),
                "timestamp": meta.get("timestamp"),
            }
        )

    merged = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    meta_summary = {
        "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": prompt_entries,
        "output_csv": str(args.output),
        "rows": int(merged.shape[0]),
    }
    args.meta_output.parent.mkdir(parents=True, exist_ok=True)
    args.meta_output.write_text(json.dumps(meta_summary, indent=2) + "\n", encoding="utf-8")

    print(f"Merged {len(frames)} files into {args.output} ({merged.shape[0]} rows)")


if __name__ == "__main__":
    main()
