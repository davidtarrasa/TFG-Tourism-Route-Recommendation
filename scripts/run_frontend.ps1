param(
  [int]$Port = 8081
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m http.server $Port
