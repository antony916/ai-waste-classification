# ♻️ AI-Based Waste Classification System

An image-classification application using MobileNetV3-Large transfer learning and Streamlit to screen **14 waste categories**.

## 14 Classes
Cardboard, Glass, Metal, Paper, Plastic, Trash, Food/Vegetable Waste, Fruit Waste, Leaves/Organic Waste, Clothes/Textile, Batteries, Electronics/E-Waste, Wood, Chemical Waste.

## Dataset
The expanded pipeline combines TrashNet, the RealWaste-derived Trash Optimizer dataset, the cleaned Hugging Face Waste Classification dataset, and genuine fruit-waste imagery from the public `waste_pictures` dataset. Fruit waste is deliberately not learned from fresh-fruit recognition images.

## Architecture
- Python 3.12
- PyTorch + torchvision
- MobileNetV3-Large pretrained weights
- 224 × 224 input
- Reproducible stratified 80/10/10 train/validation/test split
- AdamW + class-weighted cross-entropy
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

The first run prepares the expanded 14-class dataset. If automatic Kaggle access is unavailable, provide genuine fruit-waste images under `.cache/waste14/waste_pictures/fruit_waste/`.

## Evaluate
```powershell
$env:PYTHONPATH="."
python src/evaluate.py
```

Evaluation creates accuracy, balanced accuracy, weighted/macro precision, recall and F1, a per-class report and a 14×14 confusion matrix in outputs/. Do not publish a final accuracy until evaluation has actually been run.

## Run
```powershell
streamlit run app.py
```

The app shows a visible checking/preprocessing/AI-analysis sequence for every new image, then displays prediction, confidence, top-5 predictions and category-specific general disposal guidance for all 14 classes.

## Important limitation
The previous 92.49% test accuracy belongs to the old 6-class model and is not a metric for the new 14-class system. New metrics must be generated after retraining.

## Limitation
This is an academic screening prototype. Disposal rules vary by municipality and material condition; verify locally.

## Status
Technical implementation prepared. Final metrics, screenshots, report and PPT updates will use actual evaluation results.
