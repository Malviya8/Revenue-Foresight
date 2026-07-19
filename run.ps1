# Windows helper mirroring run.sh for local scoring tests.
param(
    [string]$DataDir = "./data",
    [string]$ModelPath = "./pickle/model.pkl",
    [string]$OutputPath = "./output/predictions.csv"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $DataDir -PathType Container)) {
    throw "DATA_DIR is not a directory: $DataDir"
}
if (-not (Test-Path $ModelPath -PathType Leaf)) {
    throw "MODEL_PATH not found: $ModelPath"
}

$OutDir = Split-Path -Parent $OutputPath
if ([string]::IsNullOrWhiteSpace($OutDir)) { $OutDir = "." }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$FeaturesPath = Join-Path $OutDir "features.parquet"
$ReportPath = Join-Path $OutDir "reconcile_report.json"

python src/generate_features.py --data-dir $DataDir --out $FeaturesPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python src/predict.py --features $FeaturesPath --model $ModelPath --output $OutputPath --report-out $ReportPath
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Output "Done. Predictions written to $OutputPath"
