# Training Guide — AI Waste Classification

## Environment
Use Python 3.12 and install requirements.

```powershell
pip install -r requirements.txt
```

## Expanded 14-class dataset
The training pipeline combines public waste-image sources for cardboard, glass, metal, paper, plastic, trash, food/vegetable waste, leaves/organic waste, clothes, batteries, electronics, wood and chemical waste. Genuine fruit-waste imagery is required for the `fruit_waste` class.

The first training run creates a cached dataset under `.cache/waste14/`.

If automatic Kaggle access is unavailable, provide genuine fruit-waste images under:

`.cache/waste14/waste_pictures/fruit_waste/`

At least 30 fruit-waste images are required before training continues.

## Train
```powershell
$env:PYTHONPATH="."
python src/train.py
```

The script creates a stratified 80/10/10 split, applies augmentation, uses class-weighted cross-entropy, fine-tunes MobileNetV3-Large, uses CUDA when available, and saves the best validation checkpoint.

## Evaluate
```powershell
$env:PYTHONPATH="."
python src/evaluate.py
```

This reports accuracy, balanced accuracy, weighted/macro precision, weighted/macro recall, weighted/macro F1 and a 14×14 confusion matrix.

## Run the app
```powershell
streamlit run app.py
```

The Streamlit app accepts the trained 14-class artifacts and displays the predicted category, confidence, top-5 predictions and general disposal guidance.

## Academic reporting
Only use metrics printed by `evaluate.py` in the final report and presentation. The previous 92.49% result belongs to the old 6-class model and must not be reused for the expanded system.
