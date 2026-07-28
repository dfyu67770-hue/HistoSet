# HistoSet

HistoSet is an uncertainty-aware, concept-based framework for prostate cancer histopathology segmentation and Gleason-pattern evaluation. The project combines hierarchical morphologic concepts, soft-label supervision, direct Gleason-pattern prediction, conformal concept sets, and rare-concept sensitivity analysis.

This repository contains the release materials associated with the HistoSet manuscript package prepared for Journal of Translational Medicine submission.

## Repository Contents

- `manuscript_package/`: manuscript-facing release candidate with main figures, supplementary figures, source data, supplementary tables, manuscript document, cover letter, and checksums.
- `scripts/`: reproducibility notes and code-entry documentation.
- `docs/`: release workflow and data-use notes.
- `LICENSE`: MIT software licence for author-developed HistoSet code.
- `CITATION.cff`: citation metadata for GitHub and Zenodo.
- `.zenodo.json`: Zenodo metadata template for archived releases.

## Data Boundary

This repository does not redistribute third-party source histopathology images or masks. Users should obtain source images and annotations from the original public resources under their respective access conditions. The included source-data CSV files support figure-level verification and manuscript review.

## Release

The intended first archival release is `v1.0.0`. After this repository is connected to Zenodo and a GitHub release is published, the Zenodo DOI should be added to the manuscript Data and Code Availability statement and to this README.

## Licence

Author-developed HistoSet code and documentation are released under the MIT License. Third-party datasets remain under their original access terms and licences.
