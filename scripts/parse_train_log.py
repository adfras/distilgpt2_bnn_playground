#!/usr/bin/env python
"""Parse training logs to produce metrics JSON compliant with Agent.MD."""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def extract_json_blocks(lines: List[str]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("{"):
            brace_level = 0
            buffer: list[str] = []
            while i < len(lines):
                buffer.append(lines[i])
                brace_level += lines[i].count("{")
                brace_level -= lines[i].count("}")
                i += 1
                if brace_level == 0:
                    text = "".join(buffer)
                    try:
                        blocks.append(json.loads(text))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Failed to decode JSON block: {exc}") from exc
                    break
            continue
        i += 1
    return blocks


def parse_loaded_line(lines: list[str]) -> Dict[str, Any]:
    pattern = re.compile(
        r"Loaded\\s+(?P<num>\\d+)\\s+examples,\\s+hidden_dim=(?P<hid>\\d+),\\s+"
        r"threshold=(?P<thresh>[-+0-9.eE]+),\\s+positive_frac=(?P<frac>[-+0-9.eE]+)"
    )
    for line in lines:
        match = pattern.search(line)
        if match:
            return {
                "num_examples": int(match.group("num")),
                "hidden_dim": int(match.group("hid")),
                "threshold": float(match.group("thresh")),
                "positive_frac": float(match.group("frac")),
            }
    return {
        "num_examples": None,
        "hidden_dim": None,
        "threshold": None,
        "positive_frac": None,
    }


def parse_metrics(lines: list[str]) -> Dict[str, float]:
    auc_pattern = re.compile(r"AUC[^:]*:\s*([-+0-9.]+)")
    brier_pattern = re.compile(r"Brier[^:]*:\s*([-+0-9.]+)")

    auc = None
    brier = None
    for line in lines:
        if auc is None:
            match = auc_pattern.search(line)
            if match:
                auc = float(match.group(1))
        if brier is None:
            match = brier_pattern.search(line)
            if match:
                brier = float(match.group(1))
        if auc is not None and brier is not None:
            break

    if auc is None or brier is None:
        raise ValueError("Failed to parse AUC and Brier from log")
    return {"auc": auc, "brier": brier}


def load_prompt_hash(features_csv: Path, override_meta: Path | None, config_meta: str | None) -> str | None:
    if override_meta is not None:
        meta_path = override_meta
    elif config_meta:
        meta_path = Path(config_meta)
    else:
        meta_path = features_csv.with_suffix(".meta.json")

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return meta.get("prompt_hash")
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Failed to read metadata file {meta_path}: {exc}") from exc
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse training log to metrics JSON")
    parser.add_argument("log_path", type=Path, help="Path to train_log.txt produced by tee")
    parser.add_argument("--method", required=True, choices=["bayesian_vi", "logistic_baseline", "lm_baseline"])
    parser.add_argument("--output", type=Path, required=True, help="Output metrics JSON path")
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Override metadata JSON path (otherwise derived from config/features path)",
    )
    args = parser.parse_args()

    lines = args.log_path.read_text(encoding="utf-8").splitlines()

    blocks = extract_json_blocks(lines)
    if not blocks:
        raise ValueError("Could not find any JSON blocks in log")

    config = blocks[0]

    metrics_text = parse_metrics(lines)
    example_meta = parse_loaded_line(lines)

    tail_metrics = blocks[-1] if len(blocks) > 1 else blocks[0]
    payload: Dict[str, Any] = dict(tail_metrics)

    if "auc" not in payload:
        payload["auc"] = metrics_text.get("auc")
    if "brier" not in payload:
        payload["brier"] = metrics_text.get("brier")

    features_entry = config.get("features_csv")
    if isinstance(features_entry, list):
        features_csv = [str(Path(p)) for p in features_entry]
        first_feature = Path(features_entry[0])
    else:
        features_csv = str(features_entry) if features_entry is not None else None
        first_feature = Path(features_entry) if features_entry else args.log_path

    prompt_hash = load_prompt_hash(first_feature, args.meta_path, config.get("meta_path"))
    if prompt_hash is None:
        prompt_hash = payload.get("prompt_hash")

    payload.setdefault("method", args.method)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    payload.setdefault("features_csv", features_csv)
    payload.setdefault("label_column", config.get("label_column"))
    payload.setdefault("seed", config.get("seed"))
    payload.setdefault("split_seed", config.get("split_seed"))
    payload.setdefault("test_prompts", config.get("test_prompts"))
    payload["prompt_hash"] = prompt_hash
    payload["log_path"] = str(args.log_path)


    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
