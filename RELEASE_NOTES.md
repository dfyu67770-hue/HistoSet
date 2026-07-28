# Release Notes

## v1.1.2

DOI metadata workflow release.

- Updates the DOI insertion workflow for README, `CITATION.cff`, `.zenodo.json`, and manuscript Data and Code Availability text.

## v1.1.1

Reproducibility hardening release.

- Adds `requirements.txt`, `requirements-dev.txt`, `requirements-dl.txt`, `requirements-lock.cpu.txt`, and `environment.yml`.
- Extends release QA to validate main figures, supplementary figures, source-data CSVs, supplementary tables, and public text hygiene.
- Adds tests for release QA behavior.
- Updates the release workflow with compile, QA, and test commands.

## v1.1.0

Training-pipeline release.

- Adds PyTorch dataset classes for prepared HistoSet NPZ tensors.
- Adds a dual-head HistoSet U-Net baseline with explanation and direct Gleason-pattern heads.
- Adds soft-label, class-balanced, Dice, hierarchy-consistency, and tumor-margin losses.
- Converts `scripts/train.py`, `scripts/test.py`, and `scripts/run_histoset.py` into executable training, evaluation, and prediction entry points.
- Adds PyTorch component tests for model and hierarchy-loss behavior.

## v1.0.2

Reproducibility workflow release.

- Adds configuration files for data, augmentation, model, loss, optimization, trainer, and conformal calibration settings.
- Adds setup, prediction, evaluation, training, and testing entry-point scripts.
- Adds `.env.example`, `.python-version`, and `CHANGELOG.md`.
- Updates README structure to document system requirements, data setup, prediction, visualization, and model training.

## v1.0.1

Repository-structure and metadata release.

- Replaces submission-stage wording with neutral study and reproducibility language.
- Adds software-style setup, prediction, visualization, and training entry points.
- Adds configuration scaffolds for the HistoSet workflow.

## v1.0.0

Initial HistoSet release package.

- Provides the figure package, supplementary tables, source-data CSVs, manuscript document, and checksum manifest.
- Defines MIT licensing for author-developed HistoSet code and documentation.
- Includes Zenodo metadata for software archiving.
- Does not redistribute third-party source histopathology images or masks.
