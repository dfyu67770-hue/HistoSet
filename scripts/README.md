# Reproducibility Scripts

The manuscript package includes source-data CSV files and checksums for figure-level verification. Full model training requires third-party histopathology images and annotations obtained from the original public resources under their respective access conditions.

The public code release should expose these reproducibility stages:

1. Data inventory and manifest construction.
2. Label harmonization.
3. HistoSet training and model selection.
4. Conformal calibration and evaluation.
5. Main and supplementary figure generation.
6. Manuscript package assembly and checksum export.

Large source images, masks, local checkpoints, and derived tensor caches should not be committed to GitHub.
