# HistoSet

Version: 1.1.2

HistoSet is an uncertainty-aware, concept-based framework for prostate cancer histopathology segmentation and Gleason-pattern evaluation. The project combines hierarchical morphologic concepts, soft-label supervision, direct Gleason-pattern prediction, conformal concept sets, and rare-concept sensitivity analysis.

Repository: https://github.com/dfyu67770-hue/HistoSet

## Paper

The manuscript and figure package are distributed in `manuscript_package/`. A Zenodo DOI will be added after archival deposition.

## System Requirements

The project was created with Python 3.10 or newer. Core utilities require only NumPy and pandas. Full model training and figure regeneration require the additional packages listed in `pyproject.toml` and access to the public histopathology datasets described below.

The code was prepared and tested on Windows with the bundled Codex Python runtime. GPU acceleration is recommended for full segmentation model training.

## Use

### General set up

Create and activate a virtual environment, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Alternative environment files are included for reproducibility:

```bash
pip install -r requirements-dev.txt
pip install -r requirements-dl.txt
```

Conda users can start from:

```bash
conda env create -f environment.yml
conda activate histoset
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Set the following environment variables before running the full workflow:

- `HISTOSET_DATASET_LOCATION`: directory containing third-party image and annotation resources.
- `HISTOSET_EXPERIMENT_LOCATION`: directory for checkpoints, logs, metrics, and derived tensors.

An example environment file is provided in `.env.example`.

### Data set up

Third-party histopathology images and masks are not redistributed in this repository. Users should obtain the source data from the cited public resources and place them under `HISTOSET_DATASET_LOCATION`.

The expected local data structure is:

```text
[HISTOSET_DATASET_LOCATION]/
  HistoSet/
    tissuearray/
      images/
      masks/
      annotations/
    gleason2019/
      images/
      maps/
    harvard_ocymp/
      images/
      annotations/
```

Run the setup entry point to validate paths and write a manifest:

```bash
python scripts/setup.py --data-root "[HISTOSET_DATASET_LOCATION]/HistoSet" --output manifest/histoset_manifest.csv
```

If prepared NPZ tensors are already available, pass their manifest directly to
the training and evaluation scripts.

### Image predictions

Run HistoSet prediction on one image or an image directory:

```bash
python scripts/run_histoset.py --images "/path/to/prepared_npz_or_directory" --checkpoint "/path/to/checkpoint.pt" --save-path "/path/to/output"
```

The prediction output includes explanation probabilities, explanation-derived Gleason-pattern probabilities, and direct Gleason-pattern probabilities when the checkpoint contains a direct pattern head.

### Paper visualizations

The released source-data CSV files and figure files can be verified with:

```bash
python scripts/evaluate_paper_results.py \
  --source-data manuscript_package/source_data \
  --figure-dir manuscript_package/figures \
  --supplementary-table-dir manuscript_package/supplementary_tables \
  --repo-root .
```

### Model training

Training uses the configuration files in `configs/`. Example calls:

```bash
python scripts/train.py --manifest "/path/to/prepared_manifest.csv" --output-dir "/path/to/experiment" --mode histoset_hierarchy
python scripts/test.py --manifest "/path/to/prepared_manifest.csv" --checkpoint "/path/to/checkpoint.pt" --output-dir "/path/to/evaluation" --mode histoset_hierarchy
```

The default configuration files expose the data loading, augmentation, model, loss, optimization, trainer, and conformal calibration settings used by the HistoSet workflow. Command-line arguments override these defaults in the lightweight public entry points.

## Repository Contents

- `configs/`: configuration files for data loading, augmentation, model, losses, optimization, trainer, and conformal calibration.
- `histoset/`: author-developed HistoSet utility package.
- `scripts/`: setup, training, testing, prediction, and figure-regeneration entry points.
- `tests/`: unit tests for hierarchy, metrics, and conformal utilities.
- `manuscript_package/`: figure files, source-data tables, supplementary tables, manuscript document, and checksums.
- `docs/`: release workflow and data-use notes.
- `LICENSE`: MIT software licence for author-developed HistoSet code.
- `CITATION.cff`: citation metadata for GitHub and Zenodo.
- `.zenodo.json`: Zenodo metadata template for archived releases.
- `.env.example`: local environment-variable template.

## Data Boundary

This repository does not redistribute third-party source histopathology images or masks. Users should obtain source images and annotations from the cited public resources under their respective access conditions. The included source-data CSV files support figure-level verification and manuscript review.

## Release

The current GitHub release target is `v1.1.2`: https://github.com/dfyu67770-hue/HistoSet/releases/tag/v1.1.2. After Zenodo archiving is completed, the Zenodo DOI should be added to the Data and Code Availability statement, `CITATION.cff`, and this README.

## Licence

Author-developed HistoSet code and documentation are released under the MIT License. Third-party datasets remain under their source access terms and licences.
