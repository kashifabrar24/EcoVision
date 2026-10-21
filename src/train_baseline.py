"""Baseline CNN training pipeline for EcoVision waste image classification.

Trains a 4-block CNN from scratch with class-weighted CrossEntropyLoss,
ImageNet normalization, Adam optimizer (lr=1e-3), batch size 32,
and early stopping on validation macro-F1 (patience=7).
Evaluates strictly on the validation set — never touches the test set.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import BaselineCNN


def seed_everything(seed: int = 42) -> None:
    """Ensure complete reproducibility across python, numpy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for a single epoch and return average training loss."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    return running_loss / total_samples


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Evaluate on a validation dataloader, returning loss, targets, and predictions."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            preds = torch.argmax(outputs, dim=1)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    avg_loss = running_loss / total_samples
    return avg_loss, np.array(all_targets), np.array(all_preds)


def main():
    print("=" * 60)
    print("EcoVision - Phase 3: Baseline CNN Training (from scratch)")
    print("=" * 60)

    # 1. Hardware & Reproducibility Setup (Rule 7 & Rule 3)
    if not torch.cuda.is_available():
        raise RuntimeError("Rule 7 Violation: CUDA is not available! Training must be on GPU.")
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")

    seed_everything(42)

    # Paths
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 2. Datasets & DataLoaders (Train and Val ONLY, NEVER touch Test)
    # ImageNet normalization, NO augmentation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_path = data_dir / "train"
    val_path = data_dir / "val"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Processed train/val directories missing under {data_dir}. Run dedupe_split.py first.")

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    val_dataset = datasets.ImageFolder(val_path, transform=transform)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Loaded datasets: {len(train_dataset)} train images, {len(val_dataset)} val images across {num_classes} classes.")
    print(f"Classes: {class_names}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 3. Class-Weighted Loss (Computed from TRAIN split only)
    train_targets = [lbl for _, lbl in train_dataset.samples]
    train_counts = torch.bincount(torch.tensor(train_targets), minlength=num_classes).float()
    total_train = len(train_targets)
    # Balanced weights: N / (num_classes * count)
    class_weights = total_train / (num_classes * train_counts)
    class_weights = class_weights.to(device)

    weight_info = {cls: round(w.item(), 4) for cls, w in zip(class_names, class_weights)}
    print(f"Computed train class weights (balanced): {weight_info}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Model Architecture & Optimizer
    model = BaselineCNN(num_classes=num_classes, dropout_p=0.3).to(device)
    total_params = model.get_num_parameters()
    print(f"Model: BaselineCNN | Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training Loop with Early Stopping (patience 7 on Validation Macro-F1)
    max_epochs = 40
    patience = 7
    best_val_macro_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    checkpoint_path = models_dir / "baseline_cnn.pt"

    history = []
    start_time = time.time()

    print(f"Starting training: max {max_epochs} epochs, Adam(lr=1e-3), patience={patience} on Val Macro-F1...")
    print("-" * 80)

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_targets, val_preds = evaluate(model, val_loader, criterion, device)

        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        epoch_sec = time.time() - epoch_start

        # Check early stopping condition
        is_best = val_macro_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            status_tag = f"[BEST - saved to {checkpoint_path.name}]"
        else:
            patience_counter += 1
            status_tag = f"(patience {patience_counter}/{patience})"

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "epoch_time_sec": round(epoch_sec, 2),
        })

        # Exactly one clean line per epoch
        print(
            f"Epoch [{epoch:02d}/{max_epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Macro-F1: {val_macro_f1:.4f} | "
            f"Time: {epoch_sec:.1f}s {status_tag}"
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement in Val Macro-F1 for {patience} epochs).")
            break

    total_training_time = time.time() - start_time
    print("-" * 80)
    print(f"Training Complete! Total Time: {total_training_time:.2f}s ({total_training_time / 60:.2f} min)")
    print(f"Best Validation Macro-F1: {best_val_macro_f1:.4f} achieved at Epoch {best_epoch}")

    # 6. Save Training History
    history_df = pd.DataFrame(history)
    history_path = reports_dir / "baseline_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"[SAVED] Per-epoch training history saved to {history_path}")

    # 7. Final Validation Evaluation on Best Checkpoint
    print(f"Loading best checkpoint from {checkpoint_path} for final validation evaluation...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    final_val_loss, final_val_targets, final_val_preds = evaluate(model, val_loader, criterion, device)

    final_val_acc = accuracy_score(final_val_targets, final_val_preds)
    final_val_macro_f1 = f1_score(final_val_targets, final_val_preds, average="macro", zero_division=0)
    report = classification_report(
        final_val_targets,
        final_val_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_metrics = {}
    for cls in class_names:
        per_class_metrics[cls] = {
            "precision": round(report[cls]["precision"], 4),
            "recall": round(report[cls]["recall"], 4),
            "f1_score": round(report[cls]["f1-score"], 4),
            "support": int(report[cls]["support"]),
        }

    metrics_payload = {
        "model": "BaselineCNN",
        "num_parameters": total_params,
        "best_epoch": best_epoch,
        "total_epochs_trained": len(history_df),
        "total_training_time_seconds": round(total_training_time, 2),
        "val_loss": round(final_val_loss, 4),
        "val_accuracy": round(final_val_acc, 4),
        "val_macro_f1": round(final_val_macro_f1, 4),
        "class_weights": weight_info,
        "per_class": per_class_metrics,
    }

    metrics_path = reports_dir / "baseline_val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[SAVED] Validation metrics saved to {metrics_path}")

    # 8. Plot Training Curves (Loss and Macro-F1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = history_df["epoch"]
    ax1.plot(epochs_range, history_df["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs_range, history_df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax1.set_title("CrossEntropy Loss over Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    ax2.plot(epochs_range, history_df["val_macro_f1"], label="Val Macro-F1", color="#2ca02c", linewidth=2)
    ax2.plot(epochs_range, history_df["val_accuracy"], label="Val Accuracy", color="#d62728", linewidth=1.5, linestyle=":")
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({best_epoch})")
    ax2.scatter(
        [best_epoch],
        [history_df.loc[history_df['epoch'] == best_epoch, 'val_macro_f1'].values[0]],
        color="#2ca02c",
        s=80,
        zorder=5,
        label=f"Best Macro-F1 ({best_val_macro_f1:.4f})"
    )
    ax2.set_title("Validation Macro-F1 & Accuracy over Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("EcoVision Baseline CNN - Training & Validation Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    curves_path = reports_dir / "baseline_curves.png"
    plt.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Curves plot saved to {curves_path}")

    # Summary Table
    print("\n" + "=" * 60)
    print("Baseline CNN Validation Performance Summary:")
    print("=" * 60)
    print(f"Overall Accuracy:  {final_val_acc * 100:.2f}%")
    print(f"Macro-F1 Score:    {final_val_macro_f1:.4f}")
    print("-" * 60)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 60)
    for cls in class_names:
        m = per_class_metrics[cls]
        print(f"{cls:<12} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['support']:<8d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
