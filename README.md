# ♻️ AI-Based Waste Classification System

An image-classification application that identifies common waste categories using MobileNetV3-Large transfer learning and Streamlit.

## Classes
Cardboard, Glass, Metal, Paper, Plastic, Trash.

## Dataset
TrashNet (garythung/trashnet) through Hugging Face Datasets.

## Architecture
- Python 3.12
- PyTorch + torchvision
- MobileNetV3-Large pretrained weights
- 224 × 224 input
- Reproducible 80/10/10 train/validation/test split
- AdamW + cross-entropy
- Training augmentation
- CUDA when an NVIDIA GPU is available
- Streamlit application

## Structure
```
ai-waste-classification/
├── ACADEMIC_MATERIALS/
├── artifacts/
├── outputs/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
├── app.py
├── config.py
├── requirements.txt
├── README.md
└── TRAINING_GUIDE.md
```

## Train
```powershell
$env:PYTHONPATH="."
python src/train.py
```

## Evaluate
```powershell
$env:PYTHONPATH="."
python src/evaluate.py
```

Evaluation creates classification metrics and a confusion matrix in outputs/. Do not publish a final accuracy until evaluation has actually been run.

## Run
```powershell
streamlit run app.py
```

The app shows a visible checking/preprocessing/AI-analysis sequence for every new image, then displays prediction, confidence, top-5 predictions and general disposal guidance.

## Limitation
This is an academic screening prototype. Disposal rules vary by municipality and material condition; verify locally.

## Status
Technical implementation prepared. Final metrics, screenshots, report and PPT updates will use actual evaluation results.
