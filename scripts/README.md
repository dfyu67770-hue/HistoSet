# Reproducibility Scripts

The release includes source-data CSV files and checksums for figure-level verification. Full model training requires third-party histopathology images and annotations obtained from the cited public resources under their respective access conditions.

The public code release exposes these reproducibility stages:

1. `setup.py`: data inventory and manifest construction for local image resources.
2. `train.py`: HistoSet model training on prepared NPZ tensors.
3. `test.py`: checkpoint evaluation on validation and test splits.
4. `run_histoset.py`: prediction export for prepared NPZ tensors.
5. `evaluate_paper_results.py`: released source-data, figure-package, supplementary-table, and public text-hygiene verification.

Large source images, masks, local checkpoints, and derived tensor caches should not be committed to GitHub.
