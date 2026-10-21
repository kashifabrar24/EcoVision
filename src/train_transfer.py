"""Phase 4: Transfer Learning with MobileNetV2 for EcoVision.

Two-stage transfer learning pipeline:
  Stage 1: Frozen backbone, train new classification head (Dropout 0.3 + Linear 1280->6).
           Adam lr 1e-3, max 10 epochs, early stopping patience 3.
  Stage 2: Fine-tune last 3 inverted-residual blocks (features[-3:]) with lr 1e-4
           and head with lr 1e-3. Frozen BatchNorm blocks kept in eval mode.
           Max 20 epochs, early stopping patience 5.

Strictly evaluates on validation data. Never touches the test set.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import BaselineCNN, create_mobilenetv2_transfer


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


def train_one_epoch_transfer(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage: int,
) -> float:
    """Train for one epoch, keeping BatchNorm layers of frozen blocks in eval mode."""
    model.train()

    # Rule: Keep BatchNorm layers of frozen blocks in eval mode!
    if stage == 1:
        # In Stage 1, all features are frozen
        for m in model.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    elif stage == 2:
        # In Stage 2, features[:-3] are frozen
        for m in model.features[:-3].modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

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
    """Evaluate on dataloader and return avg loss, targets array, and preds array."""
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

    return running_loss / total_samples, np.array(all_targets), np.array(all_preds)


def main():
    print("=" * 65)
    print("EcoVision - Phase 4: Transfer Learning (MobileNetV2 ImageNet)")
    print("=" * 65)

    # 1. Hardware & Seed Verification
    if not torch.cuda.is_available():
        raise RuntimeError("Rule 7 Violation: CUDA is not available! Must train on GPU.")
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")

    seed_everything(42)

    # 2. Paths & Data Setup (Train and Val ONLY, NEVER touch Test)
    data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=transform)
    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Datasets: {len(train_dataset)} train images, {len(val_dataset)} val images across {num_classes} classes.")

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

    # 3. Class-Weighted Loss (TRAIN split only)
    train_targets = [lbl for _, lbl in train_dataset.samples]
    train_counts = torch.bincount(torch.tensor(train_targets), minlength=num_classes).float()
    total_train = len(train_targets)
    class_weights = total_train / (num_classes * train_counts)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Model Instantiation
    model = create_mobilenetv2_transfer(num_classes=num_classes, dropout_p=0.3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated MobileNetV2 Transfer Model. Total Parameters: {total_params:,}")

    history = []
    checkpoint_best_path = models_dir / "mobilenetv2_transfer.pt"
    stage1_best_path = models_dir / "temp_mobilenetv2_stage1_best.pt"

    start_total_time = time.time()
    best_overall_macro_f1 = -1.0
    best_overall_epoch = -1
    best_overall_stage = -1

    # -------------------------------------------------------------
    # STAGE 1: Freeze Backbone, Train Head Only
    # -------------------------------------------------------------
    print("\n" + "-" * 65)
    print("STAGE 1: Feature Extraction (Freeze backbone, train head only)")
    print("Config: Adam(lr=1e-3), Max 10 epochs, Early Stopping Patience=3")
    print("-" * 65)

    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    stage1_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1 Trainable Parameters: {stage1_trainable_params:,} (classifier head only)")

    optimizer_stage1 = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    max_epochs_stage1 = 10
    patience_stage1 = 3
    best_s1_f1 = -1.0
    best_s1_epoch = -1
    patience_cnt_s1 = 0

    for epoch in range(1, max_epochs_stage1 + 1):
        ep_start = time.time()
        t_loss = train_one_epoch_transfer(model, train_loader, criterion, optimizer_stage1, device, stage=1)
        v_loss, v_targets, v_preds = evaluate(model, val_loader, criterion, device)

        v_acc = accuracy_score(v_targets, v_preds)
        v_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)
        ep_dur = time.time() - ep_start

        is_best_s1 = v_f1 > best_s1_f1
        if is_best_s1:
            best_s1_f1 = v_f1
            best_s1_epoch = epoch
            patience_cnt_s1 = 0
            torch.save(model.state_dict(), stage1_best_path)
            # Also update overall best
            best_overall_macro_f1 = v_f1
            best_overall_epoch = epoch
            best_overall_stage = 1
            torch.save(model.state_dict(), checkpoint_best_path)
            tag = f"[BEST Stage 1: {v_f1:.4f}]"
        else:
            patience_cnt_s1 += 1
            tag = f"(patience {patience_cnt_s1}/{patience_stage1})"

        history.append({
            "stage": 1,
            "epoch": epoch,
            "train_loss": round(t_loss, 4),
            "val_loss": round(v_loss, 4),
            "val_accuracy": round(v_acc, 4),
            "val_macro_f1": round(v_f1, 4),
            "epoch_time_sec": round(ep_dur, 2),
        })

        print(
            f"Stage 1 | Epoch [{epoch:02d}/{max_epochs_stage1:02d}] "
            f"Train Loss: {t_loss:.4f} | "
            f"Val Loss: {v_loss:.4f} | "
            f"Val Acc: {v_acc:.4f} | "
            f"Val Macro-F1: {v_f1:.4f} | "
            f"Time: {ep_dur:.1f}s {tag}"
        )

        if patience_cnt_s1 >= patience_stage1:
            print(f"Stage 1 early stopping triggered at epoch {epoch}.")
            break

    print(f"Stage 1 Finished. Best Stage 1 Val Macro-F1: {best_s1_f1:.4f} at Epoch {best_s1_epoch}")

    # -------------------------------------------------------------
    # STAGE 2: Unfreeze Last 3 Inverted Residual Blocks (features[-3:])
    # -------------------------------------------------------------
    print("\n" + "-" * 65)
    print("STAGE 2: Fine-Tuning (Unfreeze features[-3:] + head)")
    print("Config: Adam(features[-3:] lr=1e-4, head lr=1e-3), Max 20 epochs, Patience=5")
    print("-" * 65)

    # Load best Stage 1 weights
    print(f"Loading best Stage 1 checkpoint from {stage1_best_path.name}...")
    model.load_state_dict(torch.load(stage1_best_path, map_location=device))

    # Keep features[:-3] frozen, unfreeze features[-3:] and classifier
    for param in model.features[:-3].parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    stage2_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 2 Trainable Parameters: {stage2_trainable_params:,} (features[-3:] + head)")

    optimizer_stage2 = torch.optim.Adam([
        {"params": model.features[-3:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])

    max_epochs_stage2 = 20
    patience_stage2 = 5
    patience_cnt_s2 = 0

    for epoch in range(1, max_epochs_stage2 + 1):
        ep_start = time.time()
        t_loss = train_one_epoch_transfer(model, train_loader, criterion, optimizer_stage2, device, stage=2)
        v_loss, v_targets, v_preds = evaluate(model, val_loader, criterion, device)

        v_acc = accuracy_score(v_targets, v_preds)
        v_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)
        ep_dur = time.time() - ep_start

        is_best_overall = v_f1 > best_overall_macro_f1
        if is_best_overall:
            best_overall_macro_f1 = v_f1
            best_overall_epoch = epoch
            best_overall_stage = 2
            patience_cnt_s2 = 0
            torch.save(model.state_dict(), checkpoint_best_path)
            tag = f"[BEST Overall: {v_f1:.4f} - saved to {checkpoint_best_path.name}]"
        else:
            patience_cnt_s2 += 1
            tag = f"(patience {patience_cnt_s2}/{patience_stage2})"

        history.append({
            "stage": 2,
            "epoch": epoch,
            "train_loss": round(t_loss, 4),
            "val_loss": round(v_loss, 4),
            "val_accuracy": round(v_acc, 4),
            "val_macro_f1": round(v_f1, 4),
            "epoch_time_sec": round(ep_dur, 2),
        })

        print(
            f"Stage 2 | Epoch [{epoch:02d}/{max_epochs_stage2:02d}] "
            f"Train Loss: {t_loss:.4f} | "
            f"Val Loss: {v_loss:.4f} | "
            f"Val Acc: {v_acc:.4f} | "
            f"Val Macro-F1: {v_f1:.4f} | "
            f"Time: {ep_dur:.1f}s {tag}"
        )

        if patience_cnt_s2 >= patience_stage2:
            print(f"Stage 2 early stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - start_total_time
    print("-" * 65)
    print(f"Transfer Learning Complete! Total Time: {total_time:.2f}s ({total_time / 60:.2f} min)")
    print(f"Best Validation Macro-F1: {best_overall_macro_f1:.4f} (Stage {best_overall_stage}, Epoch {best_overall_epoch})")

    # Clean up temporary stage 1 checkpoint
    if stage1_best_path.exists():
        stage1_best_path.unlink()

    # 5. Save Training History
    history_df = pd.DataFrame(history)
    history_path = reports_dir / "transfer_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"[SAVED] Transfer training history saved to {history_path}")

    # 6. Final Evaluation on Best Checkpoint
    print(f"\nEvaluating final best checkpoint from {checkpoint_best_path} on validation set...")
    model.load_state_dict(torch.load(checkpoint_best_path, map_location=device))
    final_v_loss, final_v_targets, final_v_preds = evaluate(model, val_loader, criterion, device)

    final_v_acc = accuracy_score(final_v_targets, final_v_preds)
    final_v_f1 = f1_score(final_v_targets, final_v_preds, average="macro", zero_division=0)
    report = classification_report(
        final_v_targets,
        final_v_preds,
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
        "model": "MobileNetV2_Transfer",
        "total_parameters": total_params,
        "trainable_parameters_stage1": stage1_trainable_params,
        "trainable_parameters_stage2": stage2_trainable_params,
        "best_stage": best_overall_stage,
        "best_epoch": best_overall_epoch,
        "total_training_time_seconds": round(total_time, 2),
        "val_loss": round(final_v_loss, 4),
        "val_accuracy": round(final_v_acc, 4),
        "val_macro_f1": round(final_v_f1, 4),
        "per_class": per_class_metrics,
    }

    metrics_path = reports_dir / "transfer_val_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[SAVED] Transfer validation metrics saved to {metrics_path}")

    # 7. Plot Curves (Stage 1 and Stage 2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    s1_df = history_df[history_df["stage"] == 1].reset_index(drop=True)
    s2_df = history_df[history_df["stage"] == 2].reset_index(drop=True)
    s1_len = len(s1_df)

    # Global continuous step index for clean plotting
    history_df["global_step"] = range(1, len(history_df) + 1)
    split_step = s1_len + 0.5

    ax1.plot(history_df["global_step"], history_df["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(history_df["global_step"], history_df["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.axvline(split_step, color="black", linestyle="--", alpha=0.8, label="Stage Transition")
    ax1.text(s1_len / 2, ax1.get_ylim()[1] * 0.9 if ax1.get_ylim()[1] > 0 else 1.0, "Stage 1: Head", ha="center", fontsize=9, fontweight="bold", backgroundcolor="white")
    ax1.set_title("CrossEntropy Loss (Stage 1 & Stage 2)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Cumulative Step / Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    ax2.plot(history_df["global_step"], history_df["val_macro_f1"], label="Val Macro-F1", color="#2ca02c", linewidth=2)
    ax2.plot(history_df["global_step"], history_df["val_accuracy"], label="Val Accuracy", color="#d62728", linewidth=1.5, linestyle=":")
    ax2.axvline(split_step, color="black", linestyle="--", alpha=0.8, label="Stage Transition")

    # Mark overall best
    if best_overall_stage == 1:
        best_step = best_overall_epoch
    else:
        best_step = s1_len + best_overall_epoch

    best_val_row = history_df[history_df["global_step"] == best_step].iloc[0]
    ax2.scatter([best_step], [best_val_row["val_macro_f1"]], color="#2ca02c", s=100, zorder=5, label=f"Best F1 ({best_overall_macro_f1:.4f})")
    ax2.set_title("Validation Macro-F1 & Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Cumulative Step / Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", frameon=True)

    plt.suptitle("EcoVision MobileNetV2 Transfer Learning - Curves", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    curves_path = reports_dir / "transfer_curves.png"
    plt.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Transfer curves saved to {curves_path}")

    # 8. Side-by-Side Confusion Matrices (Baseline vs Transfer on Validation)
    print("\nGenerating side-by-side validation confusion matrices...")
    baseline_checkpoint = models_dir / "baseline_cnn.pt"
    if not baseline_checkpoint.exists():
        raise FileNotFoundError(f"Missing {baseline_checkpoint}. Run train_baseline.py first.")

    baseline_model = BaselineCNN(num_classes=num_classes).to(device)
    baseline_model.load_state_dict(torch.load(baseline_checkpoint, map_location=device))
    _, base_targets, base_preds = evaluate(baseline_model, val_loader, criterion, device)
    base_acc = accuracy_score(base_targets, base_preds)
    base_f1 = f1_score(base_targets, base_preds, average="macro", zero_division=0)
    cm_base = confusion_matrix(base_targets, base_preds)

    cm_transfer = confusion_matrix(final_v_targets, final_v_preds)

    fig, (ax_base, ax_trans) = plt.subplots(1, 2, figsize=(15, 6))

    sns.heatmap(
        cm_base,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax_base,
        cbar=False,
    )
    ax_base.set_title(f"Baseline CNN\nVal Acc: {base_acc*100:.2f}% | Macro-F1: {base_f1:.4f}", fontsize=11, fontweight="bold")
    ax_base.set_xlabel("Predicted Label", fontsize=10)
    ax_base.set_ylabel("True Label", fontsize=10)

    sns.heatmap(
        cm_transfer,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax_trans,
        cbar=False,
    )
    ax_trans.set_title(f"MobileNetV2 Transfer\nVal Acc: {final_v_acc*100:.2f}% | Macro-F1: {final_v_f1:.4f}", fontsize=11, fontweight="bold")
    ax_trans.set_xlabel("Predicted Label", fontsize=10)
    ax_trans.set_ylabel("True Label", fontsize=10)

    plt.suptitle("Validation Confusion Matrices Comparison (N=379)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    cm_path = reports_dir / "val_confusion_matrices.png"
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Side-by-side confusion matrices saved to {cm_path}")

    # 9. Comparison Table: Baseline vs Transfer on VALIDATION
    with open(reports_dir / "baseline_val_metrics.json", "r") as f:
        base_metrics = json.load(f)

    comparison_rows = [
        {"metric": "Overall Accuracy (%)", "baseline_cnn": f"{base_metrics['val_accuracy']*100:.2f}%", "mobilenetv2_transfer": f"{final_v_acc*100:.2f}%", "delta": f"{(final_v_acc - base_metrics['val_accuracy'])*100:+.2f}%"},
        {"metric": "Macro-F1 Score", "baseline_cnn": f"{base_metrics['val_macro_f1']:.4f}", "mobilenetv2_transfer": f"{final_v_f1:.4f}", "delta": f"{final_v_f1 - base_metrics['val_macro_f1']:+.4f}"},
        {"metric": "Cardboard F1", "baseline_cnn": f"{base_metrics['per_class']['cardboard']['f1_score']:.4f}", "mobilenetv2_transfer": f"{per_class_metrics['cardboard']['f1_score']:.4f}", "delta": f"{per_class_metrics['cardboard']['f1_score'] - base_metrics['per_class']['cardboard']['f1_score']:+.4f}"},
        {"metric": "Glass F1", "baseline_cnn": f"{base_metrics['per_class']['glass']['f1_score']:.4f}", "mobilenetv2_transfer": f"{per_class_metrics['glass']['f1_score']:.4f}", "delta": f"{per_class_metrics['glass']['f1_score'] - base_metrics['per_class']['glass']['f1_score']:+.4f}"},
        {"metric": "Metal F1", "baseline_cnn": f"{base_metrics['per_class']['metal']['f1_score']:.4f}", "mobilenetv2_transfer": f"{per_class_metrics['metal']['f1_score']:.4f}", "delta": f"{per_class_metrics['metal']['f1_score'] - base_metrics['per_class']['metal']['f1_score']:+.4f}"},
        {"metric": "Paper F1", "baseline_cnn": f"{base_metrics['per_class']['paper']['f1_score']:.4f}", "mobilenetv2_transfer": f"{per_class_metrics['paper']['f1_score']:.4f}", "delta": f"{per_class_metrics['paper']['f1_score'] - base_metrics['per_class']['paper']['f1_score']:+.4f}"},
        {"metric": "Plastic F1", "baseline_cnn": f"{base_metrics['per_class']['plastic']['f1_score']:.4f}", "mobilenetv2_transfer": f"{per_class_metrics['plastic']['f1_score']:.4f}", "delta": f"{per_class_metrics['plastic']['f1_score'] - base_metrics['per_class']['plastic']['f1_score']:+.4f}"},
        {"metric": "Trash F1", "baseline_cnn": f"{base_metrics['per_class']['trash']['f1_score']:.4f}", "mobilenetv2_transfer": f"{per_class_metrics['trash']['f1_score']:.4f}", "delta": f"{per_class_metrics['trash']['f1_score'] - base_metrics['per_class']['trash']['f1_score']:+.4f}"},
        {"metric": "Total Parameters", "baseline_cnn": f"{base_metrics['num_parameters']:,}", "mobilenetv2_transfer": f"{total_params:,}", "delta": f"{total_params - base_metrics['num_parameters']:+,}"},
        {"metric": "Trainable Parameters", "baseline_cnn": f"{base_metrics['num_parameters']:,}", "mobilenetv2_transfer": f"{stage2_trainable_params:,} (S2)", "delta": "N/A"},
        {"metric": "Training Time (s)", "baseline_cnn": f"{base_metrics['total_training_time_seconds']:.1f}s", "mobilenetv2_transfer": f"{total_time:.1f}s", "delta": f"{total_time - base_metrics['total_training_time_seconds']:+.1f}s"},
    ]

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = reports_dir / "baseline_vs_transfer_val.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"[SAVED] Comparison table saved to {comp_path}")

    # 10. Print Console Summary Table
    print("\n" + "=" * 75)
    print("VALIDATION COMPARISON: Baseline CNN vs. MobileNetV2 Transfer")
    print("=" * 75)
    print(f"{'Metric':<30} | {'Baseline CNN':<14} | {'MobileNetV2':<14} | {'Delta':<10}")
    print("-" * 75)
    for r in comparison_rows:
        print(f"{r['metric']:<30} | {r['baseline_cnn']:<14} | {r['mobilenetv2_transfer']:<14} | {r['delta']:<10}")
    print("=" * 75)


if __name__ == "__main__":
    main()
