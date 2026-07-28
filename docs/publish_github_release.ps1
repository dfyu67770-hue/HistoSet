param(
    [Parameter(Mandatory = $true)]
    [string]$Owner,

    [string]$Repo = "HistoSet",
    [string]$Visibility = "public"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI is not available. Install GitHub CLI before running this script."
}

gh auth status

$repoFullName = "$Owner/$Repo"
$archive = Join-Path (Split-Path $PSScriptRoot -Parent) "..\HistoSet-v1.0.0-release.zip"
$archive = [System.IO.Path]::GetFullPath($archive)

if (-not (Test-Path $archive)) {
    git archive --format=zip --output $archive v1.0.0
}

gh repo create $repoFullName --source . --remote origin --push --$Visibility
git push origin v1.0.0

gh release create v1.0.0 $archive `
    --repo $repoFullName `
    --title "HistoSet v1.0.0" `
    --notes-file RELEASE_NOTES.md

Write-Host "Created release: https://github.com/$repoFullName/releases/tag/v1.0.0"
