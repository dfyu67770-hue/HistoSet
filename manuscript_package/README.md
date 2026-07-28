# HistoSet manuscript release candidate

This folder contains the manuscript, figure, source-data, supplementary, and reproducibility materials for the HistoSet study.

## Contents

- `manuscript/`: editable manuscript document and PDF render.
- `cover_letter/`: journal-facing cover letter text.
- `figures/main_figures/`: main figures exported as PNG, SVG, PDF, and TIFF.
- `figures/supplementary_figures/`: supplementary figures exported as PNG, SVG, PDF, and TIFF.
- `source_data/`: machine-readable CSV source data used for quantitative figure panels.
- `supplementary_tables/`: supplementary table workbook and table CSV files.
- `docs/`: data/code availability statement.
- `manifest/`: file inventory with sizes and SHA-256 checksums.

## Data-use boundary

The package does not redistribute source histopathology images or masks. Source images and masks should be obtained from the source public resources under their respective access conditions. The included CSV files support figure-level reproducibility and manuscript review.

## Method boundary

HistoSet is intended as an uncertainty-aware, concept-based framework for transparent model evaluation. Deterministic segmentation metrics should be interpreted together with conformal coverage, set size, annotation agreement, and rare-concept sensitivity analyses.
