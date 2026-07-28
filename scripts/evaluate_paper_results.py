"""Verify released source-data tables, figure files, and public text hygiene."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


MAIN_FIGURES = {
    "figure1_fullscale_histoset_overview",
    "figure2_histoset_class_distribution",
    "figure3_histoset_image_level_agreement",
    "figure4_histoset_pixel_level_agreement",
    "figure5_histoset_hierarchy_model_results",
    "figure6_histoset_prediction_annotation_comparison",
    "figure7_histoset_case_plate",
    "figure8_histoset_explanatory_ontology",
}

SUPPLEMENTARY_FIGURES = {
    "Supplementary_Fig_S1_data_inventory",
    "Supplementary_Fig_S2_label_harmonization",
    "Supplementary_Fig_S3_split_scope",
    "Supplementary_Fig_S4_pixel_composition",
    "Supplementary_Fig_S5_model_configuration_sensitivity",
    "Supplementary_Fig_S6_seed_trajectory",
    "Supplementary_Fig_S7_replicate_stability",
    "Supplementary_Fig_S8_conformal_coverage",
    "Supplementary_Fig_S8b_alpha_sensitivity",
    "Supplementary_Fig_S9_case_selection",
    "Supplementary_Fig_S10_source_package",
    "Supplementary_Fig_S11_rare_specialist_expert_policy",
}

REQUIRED_FORMATS = {".png", ".svg", ".pdf", ".tiff"}
FORBIDDEN_TERMS = {
    "".join(chr(code) for code in [27169, 20223]),
    "".join(chr(code) for code in [23545, 26631]),
    "".join(chr(code) for code in [19968, 27604, 19968]),
    "".join(chr(code) for code in [22797, 21051]),
    "AI" + "-generated",
    "place" + "holder",
    "GleasonXAI " + "".join(["a", "l", "i", "g", "n", "e", "d"]),
    "prepared for " + "Journal",
    "Journal of Translational Medicine " + "submission",
}


def assert_nonempty_csv(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"CSV has no data rows: {path}")
    return {"rows": len(rows) - 1, "columns": len(rows[0])}


def assert_figure_set(root: Path, stems: Iterable[str], label: str) -> list[dict[str, object]]:
    records = []
    for stem in stems:
        for suffix in sorted(REQUIRED_FORMATS):
            path = root / f"{stem}{suffix}"
            if not path.exists():
                raise FileNotFoundError(f"Missing {label} file: {path}")
            size = path.stat().st_size
            if size <= 1024:
                raise ValueError(f"Figure file is unexpectedly small: {path} ({size} bytes)")
            records.append({"figure": stem, "format": suffix.lstrip("."), "bytes": size})
    return records


def scan_text_hygiene(root: Path) -> list[dict[str, str]]:
    findings = []
    allowed_suffixes = {".md", ".txt", ".py", ".toml", ".yaml", ".yml", ".cff", ".json"}
    ignored_parts = {".git", "__pycache__", ".pytest_cache"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for term in FORBIDDEN_TERMS:
            if term in text:
                findings.append({"file": str(path.relative_to(root)), "term": term})
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-data", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--supplementary-table-dir", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not args.source_data.exists():
        raise FileNotFoundError(f"No source-data CSV files found in {args.source_data}")
    if not args.figure_dir.exists():
        raise FileNotFoundError(f"No figure files found in {args.figure_dir}")

    csv_records = []
    for csv_path in sorted(args.source_data.glob("*.csv")):
        record = assert_nonempty_csv(csv_path)
        record["file"] = csv_path.name
        csv_records.append(record)
    if not csv_records:
        raise FileNotFoundError(f"No source-data CSV files found in {args.source_data}")

    main_dir = args.figure_dir / "main_figures"
    supp_dir = args.figure_dir / "supplementary_figures"
    figure_records = []
    figure_records.extend(assert_figure_set(main_dir, MAIN_FIGURES, "main figure"))
    figure_records.extend(assert_figure_set(supp_dir, SUPPLEMENTARY_FIGURES, "supplementary figure"))

    supplementary_tables = []
    if args.supplementary_table_dir is not None:
        if not args.supplementary_table_dir.exists():
            raise FileNotFoundError(f"Supplementary table directory does not exist: {args.supplementary_table_dir}")
        table_files = sorted(args.supplementary_table_dir.rglob("*.csv")) + sorted(args.supplementary_table_dir.rglob("*.xlsx"))
        if not table_files:
            raise FileNotFoundError(f"No supplementary tables found in {args.supplementary_table_dir}")
        supplementary_tables = [
            {"file": str(path.relative_to(args.supplementary_table_dir)), "bytes": path.stat().st_size}
            for path in table_files
            if path.stat().st_size > 0
        ]
        if len(supplementary_tables) != len(table_files):
            raise ValueError("One or more supplementary table files are empty")

    hygiene_findings = scan_text_hygiene(args.repo_root) if args.repo_root is not None else []
    if hygiene_findings:
        details = ", ".join(f"{item['file']}:{item['term']}" for item in hygiene_findings[:10])
        raise ValueError(f"Public text hygiene check failed: {details}")

    report = {
        "source_data_csv": len(csv_records),
        "figure_files": len(figure_records),
        "supplementary_tables": len(supplementary_tables),
        "text_hygiene_findings": hygiene_findings,
        "csv_records": csv_records,
        "figure_records": figure_records,
        "supplementary_table_records": supplementary_tables,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"source_data_csv={report['source_data_csv']}")
    print(f"figure_files={report['figure_files']}")
    print(f"supplementary_tables={report['supplementary_tables']}")
    print("text_hygiene_findings=0")


if __name__ == "__main__":
    main()
