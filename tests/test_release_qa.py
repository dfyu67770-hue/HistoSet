from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_paper_results.py"
SPEC = importlib.util.spec_from_file_location("evaluate_paper_results", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
release_qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_qa)


def test_nonempty_csv_check(tmp_path: Path):
    csv_path = tmp_path / "source.csv"
    csv_path.write_text("metric,value\nDice,0.5\n", encoding="utf-8")
    assert release_qa.assert_nonempty_csv(csv_path) == {"rows": 1, "columns": 2}


def test_empty_csv_is_rejected(tmp_path: Path):
    csv_path = tmp_path / "source.csv"
    csv_path.write_text("metric,value\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        release_qa.assert_nonempty_csv(csv_path)


def test_figure_set_requires_all_formats(tmp_path: Path):
    stem = "figure1_fullscale_histoset_overview"
    for suffix in release_qa.REQUIRED_FORMATS:
        (tmp_path / f"{stem}{suffix}").write_bytes(b"x" * 2048)
    records = release_qa.assert_figure_set(tmp_path, {stem}, "main figure")
    assert len(records) == len(release_qa.REQUIRED_FORMATS)


def test_text_hygiene_finds_process_terms(tmp_path: Path):
    bad_file = tmp_path / "README.md"
    term = "place" + "holder"
    bad_file.write_text(f"This contains {term} text.\n", encoding="utf-8")
    findings = release_qa.scan_text_hygiene(tmp_path)
    assert findings == [{"file": "README.md", "term": term}]
