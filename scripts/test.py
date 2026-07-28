"""HistoSet evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from histoset.torch_engine import build_model, evaluate_model, summarize_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", default="histoset_hierarchy")
    parser.add_argument("--split", nargs="+", default=["val", "test"])
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location="cpu")
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    model = build_model(args.mode, base_channels=args.base_channels)
    model.load_state_dict(state_dict)
    frames = [
        evaluate_model(model, args.manifest, split, args.mode, args.image_size, torch.device("cpu"))
        for split in args.split
    ]
    metrics = pd.concat(frames, ignore_index=True)
    summary = summarize_metrics(metrics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"metrics_{args.mode}.csv"
    summary_path = args.output_dir / f"metrics_summary_{args.mode}.csv"
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"metrics={metrics_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
