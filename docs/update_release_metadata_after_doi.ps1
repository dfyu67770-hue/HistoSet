param(
    [Parameter(Mandatory = $false)]
    [string]$RepositoryUrl = "https://github.com/dfyu67770-hue/HistoSet",

    [Parameter(Mandatory = $true)]
    [string]$ZenodoDoi,

    [Parameter(Mandatory = $false)]
    [string]$Version = "1.1.1"
)

$ErrorActionPreference = "Stop"

function Convert-ToDoiUrl {
    param([string]$Value)
    $clean = $Value.Trim()
    if ($clean -match "^https://doi\.org/") {
        return $clean
    }
    if ($clean -match "^doi:") {
        $clean = $clean.Substring(4).Trim()
    }
    return "https://doi.org/$clean"
}

$doiUrl = Convert-ToDoiUrl $ZenodoDoi
$doiValue = $doiUrl -replace "^https://doi\.org/", ""

$readme = Get-Content README.md -Raw
$readme = $readme -replace "A Zenodo DOI will be added after archival deposition\.", "Archived software release: $doiUrl."
$readme = $readme -replace "After Zenodo archiving is completed, the Zenodo DOI should be added to the Data and Code Availability statement, `CITATION\.cff`, and this README\.", "Archived software DOI: $doiUrl."
Set-Content README.md $readme -Encoding UTF8

$citation = Get-Content CITATION.cff
$citation = $citation | Where-Object { $_ -notmatch "^repository-code:" -and $_ -notmatch "^doi:" -and $_ -notmatch "^version:" }
$citation += "version: `"$Version`""
$citation += "repository-code: `"$RepositoryUrl`""
$citation += "doi: `"$doiValue`""
$citation | Set-Content CITATION.cff -Encoding UTF8

$zenodo = Get-Content .zenodo.json -Raw | ConvertFrom-Json
$zenodo.version = $Version
$zenodo | Add-Member -NotePropertyName doi -NotePropertyValue $doiValue -Force
$zenodo | ConvertTo-Json -Depth 10 | Set-Content .zenodo.json -Encoding UTF8

$availabilityPath = "manuscript_package\docs\data_code_availability.md"
$availability = Get-Content $availabilityPath -Raw
$availability = $availability -replace "The Zenodo archived-release DOI will be added to the final public record\.", "The archived software release is available through Zenodo at $doiUrl."
Set-Content $availabilityPath $availability -Encoding UTF8

git add README.md CITATION.cff .zenodo.json $availabilityPath
git commit -m "Add archived release DOI"

Write-Host "Updated release metadata with $doiUrl"
