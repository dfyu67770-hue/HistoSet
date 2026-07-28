param(
    [Parameter(Mandatory = $true)]
    [string]$RepositoryUrl,

    [Parameter(Mandatory = $true)]
    [string]$ZenodoDoi
)

$ErrorActionPreference = "Stop"

(Get-Content README.md) |
    ForEach-Object { $_ -replace "the Zenodo DOI should be added to the manuscript Data and Code Availability statement and to this README\.", "Archived software DOI: $ZenodoDoi." } |
    Set-Content README.md

$citation = Get-Content CITATION.cff
$citation = $citation | Where-Object { $_ -notmatch "^repository-code:" -and $_ -notmatch "^doi:" }
$citation += "repository-code: `"$RepositoryUrl`""
$citation += "doi: `"$ZenodoDoi`""
$citation | Set-Content CITATION.cff

git add README.md CITATION.cff
git commit -m "Add archived release DOI"

Write-Host "Updated README.md and CITATION.cff with $ZenodoDoi"
