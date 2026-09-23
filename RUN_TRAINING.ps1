$ErrorActionPreference = "Stop"

Write-Host "=== AI Waste Classification — 14-Class Training ===" -ForegroundColor Cyan
Write-Host "1/4 Pulling latest project code..."
git pull origin main

Write-Host "2/4 Checking Python environment..."
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Python virtual environment not found. Create .venv first."
}

$python = ".\.venv\Scripts\python.exe"
$env:PYTHONPATH = "."

Write-Host "3/4 Installing/confirming dependencies..."
& $python -m pip install -r requirements.txt

Write-Host "4/4 Starting CUDA training..."
& $python src\train.py

Write-Host "Training finished. Starting evaluation..." -ForegroundColor Green
& $python src\evaluate.py

Write-Host ""
Write-Host "=== COMPLETE ===" -ForegroundColor Green
Write-Host "Model: artifacts\waste_mobilenetv3.pth"
Write-Host "Evaluation outputs: outputs folder"
Write-Host "Evaluation outputs: outputs folder"