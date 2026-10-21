"""Phase 6: Final Test Set Evaluation for EcoVision.

Evaluates MobileNetV2 Final (from Phase 5) and Baseline CNN on the holdout test set.
The test split is evaluated here for the FIRST time.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import BaselineCNN, create_mobilenetv2_transfer


def seed_everything(seed: int = 42) -> None:
    """Ensure exact reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict_dataset(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Generate softmax probabilities and predictions for an entire dataset.

    Returns:
        targets: Array of true class indices.
        preds: Array of predicted class indices.
        probs: Array of softmax probability distributions (N, num_classes).
        filenames: List of image filenames corresponding to each sample.
    """
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False,
    )

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    filenames = [Path(path).name for path, _ in dataset.samples]
    return (
        np.array(all_targets, dtype=np.int64),
        np.array(all_preds, dtype=np.int64),
        np.array(all_probs, dtype=np.float32),
        filenames,
    )


def compute_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Compute overall accuracy, macro-F1, and per-class precision/recall/F1."""
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    clf_rep = classification_report(
        targets,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for c in class_names:
        per_class[c] = {
            "precision": round(float(clf_rep[c]["precision"]), 4),
            "recall": round(float(clf_rep[c]["recall"]), 4),
            "f1_score": round(float(clf_rep[c]["f1-score"]), 4),
            "support": int(clf_rep[c]["support"]),
        }

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
    }


def compute_bootstrap_ci(
    targets: np.ndarray,
    preds: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """Compute 95% empirical bootstrap confidence intervals for accuracy and macro-F1."""
    rng = np.random.default_rng(seed)
    n_samples = len(targets)
    boot_accs = []
    boot_macro_f1s = []

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        b_targets = targets[boot_idx]
        b_preds = preds[boot_idx]

        boot_accs.append(accuracy_score(b_targets, b_preds))
        boot_macro_f1s.append(f1_score(b_targets, b_preds, average="macro", zero_division=0))

    acc_ci = (float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5)))
    macro_f1_ci = (float(np.percentile(boot_macro_f1s, 2.5)), float(np.percentile(boot_macro_f1s, 97.5)))

    return {
        "accuracy": {
            "mean": round(float(np.mean(boot_accs)), 4),
            "ci_95_lower": round(acc_ci[0], 4),
            "ci_95_upper": round(acc_ci[1], 4),
        },
        "macro_f1": {
            "mean": round(float(np.mean(boot_macro_f1s)), 4),
            "ci_95_lower": round(macro_f1_ci[0], 4),
            "ci_95_upper": round(macro_f1_ci[1], 4),
        },
    }


def main():
    print("=" * 75)
    print("EcoVision - Phase 6: Final Holdout Test Set Evaluation")
    print("=" * 75)

    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    val_dataset = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    test_dataset = datasets.ImageFolder(data_dir / "test", transform=eval_transform)
    train_dataset = datasets.ImageFolder(data_dir / "train", transform=eval_transform)

    class_names = val_dataset.classes
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # -------------------------------------------------------------
    # 1. Load Models
    # -------------------------------------------------------------
    final_model_path = models_dir / "mobilenetv2_final.pt"
    baseline_model_path = models_dir / "baseline_cnn.pt"

    if not final_model_path.exists():
        raise FileNotFoundError(f"Missing final model: {final_model_path}")
    if not baseline_model_path.exists():
        raise FileNotFoundError(f"Missing baseline model: {baseline_model_path}")

    print(f"\nLoading MobileNetV2 Final from {final_model_path}...")
    final_model = create_mobilenetv2_transfer(num_classes=num_classes, dropout_p=0.3, pretrained=False)
    final_ckpt = torch.load(final_model_path, map_location=device)
    final_model.load_state_dict(final_ckpt)
    final_model.to(device)
    final_model.eval()

    print(f"Loading Baseline CNN from {baseline_model_path}...")
    baseline_model = BaselineCNN(num_classes=num_classes, dropout_p=0.3)
    baseline_ckpt = torch.load(baseline_model_path, map_location=device)
    baseline_model.load_state_dict(baseline_ckpt)
    baseline_model.to(device)
    baseline_model.eval()

    # -------------------------------------------------------------
    # 2. Generate Predictions
    # -------------------------------------------------------------
    print("\nEvaluating MobileNetV2 Final on validation set...")
    val_targets, val_preds, val_probs, val_filenames = predict_dataset(final_model, val_dataset, device)

    print("Evaluating MobileNetV2 Final on test set (FIRST AND ONLY EVALUATION)...")
    test_targets, test_preds, test_probs, test_filenames = predict_dataset(final_model, test_dataset, device)

    print("Evaluating Baseline CNN on test set...")
    b_test_targets, b_test_preds, b_test_probs, b_test_filenames = predict_dataset(baseline_model, test_dataset, device)

    # -------------------------------------------------------------
    # 3. Save Per-Image Predictions CSVs for Final Model
    # -------------------------------------------------------------
    val_confidences = np.max(val_probs, axis=1)
    val_df_data = {
        "filename": val_filenames,
        "true": [class_names[t] for t in val_targets],
        "pred": [class_names[p] for p in val_preds],
        "confidence": np.round(val_confidences, 6),
    }
    for i, c in enumerate(class_names):
        val_df_data[f"p_{c}"] = np.round(val_probs[:, i], 6)
    val_pred_df = pd.DataFrame(val_df_data)
    val_pred_path = reports_dir / "val_predictions.csv"
    val_pred_df.to_csv(val_pred_path, index=False)
    print(f"[SAVED] Validation predictions saved to {val_pred_path} ({len(val_pred_df)} rows)")

    test_confidences = np.max(test_probs, axis=1)
    test_df_data = {
        "filename": test_filenames,
        "true": [class_names[t] for t in test_targets],
        "pred": [class_names[p] for p in test_preds],
        "confidence": np.round(test_confidences, 6),
    }
    for i, c in enumerate(class_names):
        test_df_data[f"p_{c}"] = np.round(test_probs[:, i], 6)
    test_pred_df = pd.DataFrame(test_df_data)
    test_pred_path = reports_dir / "test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"[SAVED] Test predictions saved to {test_pred_path} ({len(test_pred_df)} rows)")

    # -------------------------------------------------------------
    # 4. Metrics & Comparison CSV / JSON
    # -------------------------------------------------------------
    final_test_metrics = compute_metrics(test_targets, test_preds, class_names)
    final_val_metrics = compute_metrics(val_targets, val_preds, class_names)
    baseline_test_metrics = compute_metrics(b_test_targets, b_test_preds, class_names)

    test_metrics_payload = {
        "mobilenetv2_final": {
            "validation": final_val_metrics,
            "test": final_test_metrics,
        },
        "baseline_cnn": {
            "test": baseline_test_metrics,
        },
    }
    metrics_json_path = reports_dir / "test_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(test_metrics_payload, f, indent=2)
    print(f"[SAVED] Metrics JSON saved to {metrics_json_path}")

    comparison_rows = [
        {
            "metric": "Overall Accuracy",
            "baseline_cnn": f"{baseline_test_metrics['accuracy']*100:.2f}%",
            "mobilenetv2_final": f"{final_test_metrics['accuracy']*100:.2f}%",
            "delta (final - baseline)": f"{(final_test_metrics['accuracy'] - baseline_test_metrics['accuracy'])*100:+.2f}%",
        },
        {
            "metric": "Macro-F1 Score",
            "baseline_cnn": f"{baseline_test_metrics['macro_f1']:.4f}",
            "mobilenetv2_final": f"{final_test_metrics['macro_f1']:.4f}",
            "delta (final - baseline)": f"{final_test_metrics['macro_f1'] - baseline_test_metrics['macro_f1']:+.4f}",
        },
    ]
    for c in class_names:
        b_f1 = baseline_test_metrics["per_class"][c]["f1_score"]
        m_f1 = final_test_metrics["per_class"][c]["f1_score"]
        comparison_rows.append({
            "metric": f"{c.capitalize()} F1",
            "baseline_cnn": f"{b_f1:.4f}",
            "mobilenetv2_final": f"{m_f1:.4f}",
            "delta (final - baseline)": f"{m_f1 - b_f1:+.4f}",
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = reports_dir / "test_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"[SAVED] Comparison CSV saved to {comparison_path}")

    # -------------------------------------------------------------
    # 5. Bootstrap Confidence Intervals
    # -------------------------------------------------------------
    final_bootstrap = compute_bootstrap_ci(test_targets, test_preds, n_bootstrap=1000, seed=42)
    baseline_bootstrap = compute_bootstrap_ci(b_test_targets, b_test_preds, n_bootstrap=1000, seed=42)

    ci_payload = {
        "mobilenetv2_final": final_bootstrap,
        "baseline_cnn": baseline_bootstrap,
    }
    ci_path = reports_dir / "test_bootstrap_ci.json"
    with open(ci_path, "w") as f:
        json.dump(ci_payload, f, indent=2)
    print(f"[SAVED] Bootstrap 95% CIs saved to {ci_path}")

    # -------------------------------------------------------------
    # 6. Confusion Matrix Visualization for Final Model
    # -------------------------------------------------------------
    cm = confusion_matrix(test_targets, test_preds)
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
        ax=ax1,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    ax1.set_title("Test Confusion Matrix (Raw Counts)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Label", fontsize=11, fontweight="bold")

    # Row-normalized (Recall)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax2,
        annot_kws={"fontsize": 11, "fontweight": "bold"},
    )
    ax2.set_title("Test Confusion Matrix (Row-Normalized / Recall)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax2.set_ylabel("True Label", fontsize=11, fontweight="bold")

    plt.suptitle("EcoVision Phase 6: MobileNetV2 Final Holdout Test Set Performance", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    cm_plot_path = reports_dir / "test_confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Test confusion matrix plot saved to {cm_plot_path}")

    # -------------------------------------------------------------
    # 7. Failure Cases & Errors CSV / Plot
    # -------------------------------------------------------------
    error_mask = (test_targets != test_preds)
    error_indices = np.where(error_mask)[0]
    total_errors = len(error_indices)
    print(f"\nTotal test misclassifications: {total_errors} / {len(test_targets)} ({total_errors/len(test_targets)*100:.2f}%)")

    test_error_df = test_pred_df[test_pred_df["true"] != test_pred_df["pred"]].copy()
    error_csv_path = reports_dir / "test_errors.csv"
    test_error_df.to_csv(error_csv_path, index=False)
    print(f"[SAVED] All test errors saved to {error_csv_path}")

    # Plot up to 16 failure cases
    num_to_plot = min(16, total_errors)
    if num_to_plot > 0:
        # Determine grid shape
        cols = 4
        rows = (num_to_plot + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3.8))
        if rows == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten()

        for idx, err_idx in enumerate(error_indices[:num_to_plot]):
            sample_path, _ = test_dataset.samples[err_idx]
            true_label = class_names[test_targets[err_idx]]
            pred_label = class_names[test_preds[err_idx]]
            conf = test_confidences[err_idx]

            img = Image.open(sample_path).convert("RGB")
            ax = axes_flat[idx]
            ax.imshow(img)
            ax.set_title(
                f"True: {true_label} | Pred: {pred_label}\nConf: {conf*100:.1f}%\n({Path(sample_path).name})",
                fontsize=9,
                fontweight="bold",
                color="#c0392b",
            )
            ax.axis("off")

        # Turn off any unused subplots
        for j in range(num_to_plot, len(axes_flat)):
            axes_flat[j].axis("off")

        plt.suptitle(
            f"EcoVision Phase 6: MobileNetV2 Final Failure Cases (Showing {num_to_plot} of {total_errors} errors)",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        failure_plot_path = reports_dir / "failure_cases.png"
        plt.savefig(failure_plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] Failure cases image grid saved to {failure_plot_path}")

    # -------------------------------------------------------------
    # 8. CHECK 1 Verifications
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("CHECK 1: Verification Results")
    print("=" * 75)

    # (a) Check test count and per-class counts against split_summary.csv
    split_summary_df = pd.read_csv(reports_dir / "split_summary.csv")
    split_test_map = dict(zip(split_summary_df["label_name"], split_summary_df["test"]))

    test_class_counts = {c: int(np.sum(test_targets == i)) for i, c in enumerate(class_names)}
    total_test_samples = len(test_targets)

    print(f"(a) Total Test Images: {total_test_samples} (Expected: {split_test_map['TOTAL']})")
    assert total_test_samples == int(split_test_map["TOTAL"]), f"Total test samples mismatch: {total_test_samples} vs {split_test_map['TOTAL']}"
    for c in class_names:
        actual_cnt = test_class_counts[c]
        expected_cnt = int(split_test_map[c])
        print(f"    - {c:<10}: {actual_cnt} (Expected: {expected_cnt})")
        assert actual_cnt == expected_cnt, f"Class {c} count mismatch: {actual_cnt} vs {expected_cnt}"
    print(">>> (a) Test sample counts match reports/split_summary.csv EXACTLY: PASS")

    # (b) No test filename appears in train or val
    train_filenames = {Path(p).name for p, _ in train_dataset.samples}
    val_filenames_set = {Path(p).name for p, _ in val_dataset.samples}
    test_filenames_set = {Path(p).name for p, _ in test_dataset.samples}

    leak_train = test_filenames_set.intersection(train_filenames)
    leak_val = test_filenames_set.intersection(val_filenames_set)
    print(f"(b) Test overlap with train: {len(leak_train)} files")
    print(f"    Test overlap with val  : {len(leak_val)} files")
    assert len(leak_train) == 0, f"Leakage detected: {leak_train}"
    assert len(leak_val) == 0, f"Leakage detected: {leak_val}"
    print(">>> (b) Zero test filename leakage into train or val: PASS")

    # (c) Recompute accuracy independently from reports/test_predictions.csv
    independent_df = pd.read_csv(test_pred_path)
    independent_acc = (independent_df["true"] == independent_df["pred"]).mean()
    print(f"(c) Reported Final Model Test Accuracy: {final_test_metrics['accuracy']:.4f}")
    print(f"    Independently recomputed accuracy : {independent_acc:.4f}")
    assert round(independent_acc, 4) == final_test_metrics["accuracy"], "Accuracy recomputation mismatch!"
    print(">>> (c) Independent accuracy recomputation matches reported metric: PASS")

    # (d) Show test macro-F1 next to the validation macro-F1 (0.9221)
    val_macro_f1_target = 0.9221
    print(f"(d) Validation Macro-F1 (Phase 5 Seed 44) : {val_macro_f1_target:.4f}")
    print(f"    Test Macro-F1 (Phase 6 Holdout)       : {final_test_metrics['macro_f1']:.4f}")
    print(f"    Delta (Test - Val)                    : {final_test_metrics['macro_f1'] - val_macro_f1_target:+.4f}")
    print(f"    Baseline CNN Test Macro-F1            : {baseline_test_metrics['macro_f1']:.4f}")

    # (e) Range check [0.85, 0.97]
    print(f"\nFinal Model Test Accuracy: {final_test_metrics['accuracy']*100:.2f}%")
    if final_test_metrics["accuracy"] < 0.85 or final_test_metrics["accuracy"] > 0.97:
        print(f"STOPPING: Test accuracy {final_test_metrics['accuracy']:.4f} is outside [0.85, 0.97] range!")
        sys.exit(1)
    else:
        print(">>> Test accuracy is comfortably within [0.85, 0.97]. Safe to proceed.")

    print("=" * 75)
    print("Phase 6 Test Evaluation Successfully Completed.")
    print("=" * 75)


if __name__ == "__main__":
    main()
