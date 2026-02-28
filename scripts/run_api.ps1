param(
  [int]$Port = 8000
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (Test-Path .\.venv\Scripts\Activate.ps1) {
  . .\.venv\Scripts\Activate.ps1
}

python -m uvicorn src.recommender.api:app --reload --port $Port
