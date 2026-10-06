# AI-Based Waste Classification System — Project Report

## 1. Project Overview

**Title:** AI-Based Waste Classification System  
**Subtitle:** An Image Classification System Using MobileNetV3-Large and Streamlit  
**Student:** Antony Paul Renio  
**Department:** Artificial Intelligence and Data Science  
**Institution:** CSI College of Engineering  
**Academic Year:** 2026–2027

This project is an end-to-end computer-vision application that classifies a user-provided waste image into one of 14 waste categories. It uses transfer learning with a pretrained MobileNetV3-Large model and provides an interactive Streamlit interface with confidence, top-5 predictions, and general disposal guidance.

## 2. Problem Statement

Manual identification of waste materials can be slow and inconsistent. The project aims to automate the first-stage identification of waste from an image so that users can receive a fast, understandable classification result.

## 3. Objectives

- Build a 14-class waste image classifier.
- Use transfer learning with MobileNetV3-Large.
- Prepare a reproducible dataset and class-wise 80/10/10 split.
- Use augmentation and class-weighted cross-entropy to improve robustness.
- Provide a Streamlit interface for image upload and prediction.
- Display the predicted category, confidence, and top-5 predictions.
- Provide general category-specific disposal guidance.
- Generate objective test metrics and a confusion matrix.

## 4. Waste Categories

1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash
7. Food / Vegetable Waste
8. Fruit Waste
9. Leaves / Organic Waste
10. Clothes / Textile
11. Batteries
12. Electronics / E-Waste
13. Wood
14. Chemical Waste

## 5. Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Vision | torchvision |
| Model | MobileNetV3-Large |
| Dataset tooling | Hugging Face datasets / Hub |
| Metrics | scikit-learn |
| Image processing | Pillow |
| Visualization | Matplotlib |
| UI | Streamlit |
| Version control | Git / GitHub |
| GPU acceleration | CUDA when available |

## 6. Dataset and Preparation

The data pipeline combines public waste-image sources and maps source labels into the project's fixed 14-class taxonomy. The preparation code validates that all required classes are represented and writes a reusable cached manifest.

The pipeline includes TrashNet, public waste-classification datasets, targeted textile/wood imagery, and fruit/vegetable waste imagery. Source labels are normalized into the project's 14 target classes.

Training uses approximately an 80/10/10 class-wise split with random seed 42.

## 7. Methodology

**Training pipeline:**

Public datasets → cleaning/class mapping → cached manifest → 80/10/10 split → augmentation → MobileNetV3-Large → class-weighted cross-entropy → AdamW → best validation model → test evaluation.

**Inference pipeline:**

User image → Streamlit upload → resize/normalize → MobileNetV3-Large → class probabilities → top prediction + confidence + top-5 → disposal guidance.

## 8. Model Configuration

| Parameter | Value |
|---|---|
| Architecture | MobileNetV3-Large |
| Input | 224 × 224 |
| Classes | 14 |
| Epochs | 15 |
| Batch size | 32 |
| Learning rate | 0.0003 |
| Optimizer | AdamW |
| Loss | Class-weighted Cross-Entropy |
| Seed | 42 |
| Split | 80% / 10% / 10% |
| Workers | 0 |
| Device | CUDA if available |

The final classifier layer is replaced with a linear layer containing 14 outputs. The best model is saved to `artifacts/waste_mobilenetv3.pth`.

## 9. Application

The Streamlit application provides:

- JPG/JPEG/PNG image upload.
- Image checking and preprocessing sequence.
- MobileNetV3-Large inference.
- Predicted waste category.
- Confidence percentage.
- Top-5 prediction breakdown.
- Category-specific general disposal guidance.
- A warning that disposal rules vary by municipality.

The application is an educational prototype and should not be treated as a certified hazardous-waste identification system.

## 10. Evaluation

The evaluation script generates:

- Accuracy
- Balanced accuracy
- Weighted precision
- Weighted recall
- Weighted F1
- Macro precision
- Macro recall
- Macro F1
- Per-class classification report
- 14 × 14 confusion matrix

Output files:

- `outputs/classification_report.txt`
- `outputs/confusion_matrix.csv`
- `outputs/confusion_matrix.png`

### Final Metrics

| Metric | Final Value |
|---|---|
| Accuracy | Pending final 14-class evaluation |
| Balanced Accuracy | Pending final 14-class evaluation |
| Weighted Precision | Pending final 14-class evaluation |
| Weighted Recall | Pending final 14-class evaluation |
| Weighted F1 | Pending final 14-class evaluation |
| Macro Precision | Pending final 14-class evaluation |
| Macro Recall | Pending final 14-class evaluation |
| Macro F1 | Pending final 14-class evaluation |

**Important:** The earlier 92.49% accuracy belonged to the previous 6-class model. It is intentionally not reported as the final result for this 14-class project.

## 11. Advantages

- Automated waste identification.
- Fourteen categories in one model.
- Transfer learning reduces training complexity.
- Confidence and top-5 predictions improve transparency.
- Interactive Streamlit interface.
- CUDA support for NVIDIA GPUs.
- Reproducible training configuration.

## 12. Limitations

- Accuracy depends on dataset quality and diversity.
- Mixed or visually ambiguous objects may be difficult to classify.
- Disposal rules vary by municipality.
- The system is an academic prototype.
- Final performance must be measured using the final 14-class test split.

## 13. Future Scope

- Add more real-world Indian waste images.
- Add object detection/segmentation for multiple objects.
- Deploy as a mobile application.
- Add regional-language guidance.
- Connect guidance to municipality-specific rules.
- Add continuous retraining.
- Add explainability such as Grad-CAM.
- Add real-time camera classification.

## 14. Conclusion

The AI-Based Waste Classification System demonstrates an end-to-end application of deep learning and computer vision for automated waste identification. It combines multi-source dataset preparation, MobileNetV3-Large transfer learning, reproducible training, objective evaluation, and a Streamlit interface.

The final system is designed to classify a user's uploaded waste image into one of fourteen categories and present the result with confidence, top-5 predictions, and general disposal guidance.

For academic submission, the final numerical metrics, confusion matrix, and application screenshots should be inserted after the final training/evaluation run.

## 15. Execution Commands

Recommended environment: Python 3.12.

```powershell
cd C:\AI_Mini_Projects\ai-waste-classification

.\.venv\Scripts\python.exe -m pip install -r requirements.txt

$env:PYTHONPATH="."

.\.venv\Scripts\python.exe src\train.py

.\.venv\Scripts\python.exe src\evaluate.py

.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 16. Final Demonstration Output

The intended final output is:

**User Image → Streamlit Upload → Resize & Normalize → MobileNetV3-Large → Softmax Probabilities → Top Prediction + Confidence + Top-5 → Disposal Guidance**

The final demonstration should capture:
1. Upload screen.
2. Uploaded waste image.
3. Predicted category.
4. Confidence.
5. Top-5 predictions.
6. Disposal guidance.
7. Evaluation classification report.
8. Confusion matrix.

---
**Report status:** Technical report prepared from the current repository. Final numerical metrics remain pending the final 14-class training and evaluation run.
