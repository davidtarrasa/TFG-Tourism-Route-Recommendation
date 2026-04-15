param(
  [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

function Keep-ReportFile([string]$relPath) {
  $r = $relPath.Replace("\", "/")

  if ($r -like "data/reports/benchmarks/*") { return $true }
  if ($r -like "data/reports/diagnostics/*") { return $true }
  if ($r -like "data/reports/maps/*") { return $true }
  if ($r -like "data/reports/routes/multi_route_q35765_latest/*") { return $true }

  if ($r -in @(
      "data/reports/etl_city_summary.csv",
      "data/reports/etl_city_summary.md",
      "data/reports/bert_category_price_labels.csv"
    )) { return $true }

  if ($r -like "data/reports/eval_*_latest.json") { return $true }
  if ($r -like "data/reports/eval_routes_*_latest.json") { return $true }
  if ($r -like "data/reports/multi_route_*_latest.json") { return $true }
  if ($r -like "data/reports/tune_all_*_latest.json") { return $true }

  return $false
}

$allFiles = Get-ChildItem "data/reports" -Recurse -File -ErrorAction SilentlyContinue
$toDelete = @()
foreach ($f in $allFiles) {
  $rel = $f.FullName.Replace($root + "\", "")
  if (-not (Keep-ReportFile -relPath $rel)) {
    $toDelete += $f.FullName
  }
}

if ($WhatIfOnly) {
  Write-Host "Dry-run. Files that would be deleted: $($toDelete.Count)"
  $toDelete | ForEach-Object { $_.Replace($root + "\", "") } | Sort-Object | Select-Object -First 200
  exit 0
}

foreach ($p in $toDelete) {
  Remove-Item -LiteralPath $p -Force
}

# Remove empty directories under reports/routes
Get-ChildItem "data/reports/routes" -Directory -ErrorAction SilentlyContinue |
  Where-Object { -not (Get-ChildItem $_.FullName -Recurse -File -ErrorAction SilentlyContinue) } |
  Remove-Item -Recurse -Force

$remaining = Get-ChildItem "data/reports" -Recurse -File -ErrorAction SilentlyContinue
$sizeBytes = ($remaining | Measure-Object Length -Sum).Sum
if (-not $sizeBytes) { $sizeBytes = 0 }

Write-Host "Deleted files: $($toDelete.Count)"
Write-Host "Remaining files: $($remaining.Count)"
Write-Host ("Remaining size MB: {0:N2}" -f ($sizeBytes / 1MB))
