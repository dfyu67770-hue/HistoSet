"""Run HistoSet inference entry point.

This lightweight public entry point validates inputs and records the requested
prediction job. Full neural-network checkpoints are not redistributed with the
source repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--save-path", required=True, type=Path)
    args = parser.parse_args()

    if not args.images.exists():
        raise FileNotFoundError(f"Image path does not exist: {args.images}")

    args.save_path.mkdir(parents=True, exist_ok=True)
    job = {
        "images": str(args.images),
        "checkpoint": str(args.checkpoint),
        "save_path": str(args.save_path),
        "status": "input_validated",
    }
    (args.save_path / "histoset_prediction_job.json").write_text(
        json.dumps(job, indent=2), encoding="utf-8"
    )
    print(f"Wrote prediction job record to {args.save_path}")


if __name__ == "__main__":
    main()
