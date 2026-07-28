"""Validate local HistoSet data paths and write a file manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def build_manifest(data_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append(
                {
                    "file_name": path.name,
                    "relative_path": path.relative_to(data_root).as_posix(),
                    "suffix": path.suffix.lower(),
                    "bytes": path.stat().st_size,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {args.data_root}")

    manifest = build_manifest(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)
    print(f"Wrote {len(manifest)} image records to {args.output}")


if __name__ == "__main__":
    main()
