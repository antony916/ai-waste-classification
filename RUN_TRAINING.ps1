$ErrorActionPreference = "Stop"
$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "Python virtual environment not found." }
$env:PYTHONPATH = "."
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
& $python src\train.py
if ($LASTEXITCODE -ne 0) { throw "Training failed." }
& $python src\evaluate.py
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed." }
Write-Host "14-class training and evaluation complete."
