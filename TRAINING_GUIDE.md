# Training Guide — AI Waste Classification

## Environment
Use Python 3.12 and install requirements.

```powershell
pip install -r requirements.txt
```

## Train
```powershell
$env:PYTHONPATH="."
python src/train.py
```

The script downloads TrashNet, creates a reproducible 80/10/10 split, applies augmentation, fine-tunes MobileNetV3-Large, uses CUDA when available, and saves the best validation checkpoint.

## Evaluate
```powershell
$env:PYTHONPATH="."
python src/evaluate.py
```

This reports accuracy, weighted precision, weighted recall and weighted F1 and creates a confusion matrix.

## Run the app
```powershell
streamlit run app.py
```

Every new upload triggers the visible checking/preprocessing/AI-analysis sequence.

## Academic reporting
Only use metrics printed by evaluate.py in the final report and presentation.
