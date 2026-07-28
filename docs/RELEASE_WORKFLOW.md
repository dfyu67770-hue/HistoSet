# GitHub and Zenodo Release Workflow

## GitHub

1. Create a public GitHub repository named `HistoSet`.
2. Push this repository to GitHub.
3. Create a semantic-versioned tag, for example `v1.1.0`.
4. Create a GitHub release from the tag.
5. Upload the manuscript release package archive if required by the journal or repository policy.

Before creating a release, run:

```powershell
python -m compileall -q histoset scripts tests
python scripts/evaluate_paper_results.py `
  --source-data manuscript_package\source_data `
  --figure-dir manuscript_package\figures `
  --supplementary-table-dir manuscript_package\supplementary_tables `
  --repo-root . `
  --output-json manuscript_package\manifest\release_qa_report.json
python -m pytest tests -q
```

If a local dependency folder adds pytest plugins that conflict with the active
environment, run the test suite with plugin autoloading disabled:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest tests -q
```

## Zenodo

1. Sign in to Zenodo with the account that should own the DOI.
2. Connect Zenodo to GitHub.
3. Enable archiving for the `HistoSet` repository.
4. Publish the selected GitHub release.
5. Record the generated Zenodo DOI in `README.md`, `CITATION.cff`, the manuscript Data Availability statement, and the final publication metadata.

## Author Fields to Finalize

- GitHub owner.
- Author names and ORCID identifiers.
- Repository URL.
- Zenodo DOI.
