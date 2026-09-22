# Training Guide

## 1. Create a Python environment
Python 3.12 is recommended.

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Train
```bash
python src/train.py
```

The script automatically uses CUDA when PyTorch detects an NVIDIA GPU.

## 4. Run the web app
```bash
streamlit run app.py
```

## 5. Academic note
Do not claim an accuracy value until the model has actually been trained and evaluated. Record the printed validation/test metrics in the report.
