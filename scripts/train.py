"""HistoSet training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config does not exist: {args.config}")
    print(f"Training configuration validated: {args.config}")
    print(f"Experiment: {args.experiment}")
    print("Full segmentation training requires local image tensors and model dependencies.")


if __name__ == "__main__":
    main()
