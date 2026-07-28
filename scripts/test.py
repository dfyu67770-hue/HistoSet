"""HistoSet evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config does not exist: {args.config}")
    print(f"Evaluation configuration validated: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
