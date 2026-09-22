import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CLASS_NAMES_PATH, MODEL_PATH, NUM_WORKERS, SEED
from src.data import WasteDataset, prepare_dataset, split_rows
from src.model import build_model


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run src/train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, class_names = prepare_dataset()
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(class_names))

    _, _, test_rows = split_rows(rows, SEED)
    loader = DataLoader(
        WasteDataset(test_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/classification_report.txt", "w", encoding="utf-8") as file:
        file.write(
            f"Test samples: {len(y_true)}\n"
            f"Accuracy: {accuracy:.4f}\n"
            f"Weighted Precision: {precision:.4f}\n"
            f"Weighted Recall: {recall:.4f}\n"
            f"Weighted F1: {f1:.4f}\n\n{report}"
        )

    np.savetxt("outputs/confusion_matrix.csv", matrix, fmt="%d", delimiter=",")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(matrix)
    ax.set_title("Waste Classification Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center")
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=180)
    plt.close(fig)

    print(f"Device: {device}")
    print(f"Test samples: {len(y_true)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1: {f1:.4f}")
    print("Saved results to: outputs")


if __name__ == "__main__":
    main()
