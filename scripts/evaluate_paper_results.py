"""Verify released source-data tables and figure files."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    args = parser.parse_args()

    csv_count = len(list(args.source_data.glob("*.csv")))
    figure_count = len(list(args.figure_dir.rglob("*.*")))
    if csv_count == 0:
        raise FileNotFoundError(f"No source-data CSV files found in {args.source_data}")
    if figure_count == 0:
        raise FileNotFoundError(f"No figure files found in {args.figure_dir}")
    print(f"source_data_csv={csv_count}")
    print(f"figure_files={figure_count}")


if __name__ == "__main__":
    main()
