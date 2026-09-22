import os
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BATCH_SIZE, CLASS_NAMES_PATH, EPOCHS, LEARNING_RATE, MODEL_PATH, NUM_WORKERS, SEED
from src.data import WasteDataset, prepare_dataset, split_rows
from src.model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main():
    set_seed(SEED)
    os.makedirs("artifacts", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Preparing TrashNet resized image dataset...")
    rows, class_names = prepare_dataset()
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
        file.write("\n".join(class_names))

    train_rows, val_rows, test_rows = split_rows(rows, SEED)
    print(f"Dataset samples: {len(rows)}")
    print(f"Split sizes: train={len(train_rows)}, validation={len(val_rows)}, test={len(test_rows)}")

    train_loader = DataLoader(
        WasteDataset(train_rows, train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        WasteDataset(val_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    best_val = -1.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved best model -> {MODEL_PATH}")

    print(f"Best validation accuracy: {best_val:.4f}")
    print("Training complete. Run src/evaluate.py for final test metrics.")


if __name__ == "__main__":
    main()
