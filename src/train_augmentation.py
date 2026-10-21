"""Phase 5: Data Augmentation Comparison Experiment for EcoVision.

Goal:
  Measure whether data augmentation improves generalization for MobileNetV2 on TrashNet,
  and explain the empirical effect (performance delta, error bars, and train/val loss gap).

Experimental Rigor:
  - Validation set only; the test set is strictly untouched.
  - Compares two configurations across seeds [42, 43, 44]:
      1. No Augmentation: ToTensor() + ImageNet normalization
      2. Augmentation: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(15),
                       ColorJitter(0.2, 0.2, 0.2), RandomResizedCrop(224, scale=(0.7, 1.0)),
                       ToTensor() + ImageNet normalization
  - Reuses Phase 4 result for No Augmentation (seed 42).
  - Keeps identical two-stage transfer recipe, class-weighted CrossEntropyLoss, and early stopping.
  - Evaluates mean +/- std over seeds, analyzes generalization loss gap, and saves mobilenetv2_final.pt.
"""

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models import create_mobilenetv2_transfer


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


def train_one_epoch_transfer(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage: int,
) -> float:
    """Train for one epoch, keeping BatchNorm layers in frozen blocks in eval mode."""
    model.train()

    # Rule: Keep BatchNorm of frozen blocks in eval mode
    if stage == 1:
        for m in model.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    elif stage == 2:
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
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate on dataloader and return avg loss, targets, and predictions."""
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


def run_experiment(
    config_name: str,
    seed: int,
    use_augmentation: bool,
    data_dir: Path,
    models_dir: Path,
    device: torch.device,
    class_names: List[str],
) -> Dict:
    """Execute the full two-stage training pipeline for a given configuration and seed."""
    seed_everything(seed)
    num_classes = len(class_names)

    # Transforms
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.ToTensor(),
            norm,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            norm,
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        norm,
    ])

    train_dataset = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir / "val", transform=val_transform)

    # Deterministic loader generator
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # Class-weighted loss derived strictly from TRAIN split
    train_targets = [lbl for _, lbl in train_dataset.samples]
    train_counts = torch.bincount(torch.tensor(train_targets), minlength=num_classes).float()
    total_train = len(train_targets)
    class_weights = total_train / (num_classes * train_counts)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Instantiate model
    model = create_mobilenetv2_transfer(num_classes=num_classes, dropout_p=0.3).to(device)

    run_ckpt_path = models_dir / f"mobilenetv2_{config_name}_seed{seed}.pt"
    temp_s1_ckpt = models_dir / f"temp_s1_{config_name}_seed{seed}.pt"

    start_run_time = time.time()
    best_overall_macro_f1 = -1.0
    best_overall_stage = -1
    best_overall_epoch = -1
    train_loss_at_best = None
    val_loss_at_best = None

    history = []

    # ---------------------------------------------------------
    # STAGE 1: Freeze Backbone, Train Head Only (Max 10, Patience 3)
    # ---------------------------------------------------------
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_s1 = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    patience_cnt_s1 = 0
    best_s1_f1 = -1.0

    for epoch in range(1, 11):
        ep_t0 = time.time()
        t_loss = train_one_epoch_transfer(model, train_loader, criterion, optimizer_s1, device, stage=1)
        v_loss, v_targets, v_preds = evaluate(model, val_loader, criterion, device)
        v_acc = accuracy_score(v_targets, v_preds)
        v_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)
        ep_dur = time.time() - ep_t0

        if v_f1 > best_s1_f1:
            best_s1_f1 = v_f1
            patience_cnt_s1 = 0
            torch.save(model.state_dict(), temp_s1_ckpt)
            best_overall_macro_f1 = v_f1
            best_overall_stage = 1
            best_overall_epoch = epoch
            train_loss_at_best = t_loss
            val_loss_at_best = v_loss
            torch.save(model.state_dict(), run_ckpt_path)
            tag = f"[BEST S1 {v_f1:.4f}]"
        else:
            patience_cnt_s1 += 1
            tag = f"(patience {patience_cnt_s1}/3)"

        history.append({
            "stage": 1, "epoch": epoch,
            "train_loss": round(t_loss, 4), "val_loss": round(v_loss, 4),
            "val_acc": round(v_acc, 4), "val_f1": round(v_f1, 4),
        })

        # Rule 7: One line per epoch
        cfg_label = "Aug" if use_augmentation else "NoAug"
        print(
            f"[{cfg_label}|S{seed}|S1-E{epoch:02d}/10] "
            f"TrainLoss: {t_loss:.4f} | ValLoss: {v_loss:.4f} | "
            f"ValAcc: {v_acc:.4f} | ValF1: {v_f1:.4f} | {ep_dur:.1f}s {tag}"
        )

        if patience_cnt_s1 >= 3:
            break

    # ---------------------------------------------------------
    # STAGE 2: Unfreeze features[-3:], lr=1e-4 & 1e-3 (Max 20, Patience 5)
    # ---------------------------------------------------------
    model.load_state_dict(torch.load(temp_s1_ckpt, map_location=device))

    for param in model.features[:-3].parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    optimizer_s2 = torch.optim.Adam([
        {"params": model.features[-3:].parameters(), "lr": 1e-4},
        {"params": model.classifier.parameters(), "lr": 1e-3},
    ])
    patience_cnt_s2 = 0

    for epoch in range(1, 21):
        ep_t0 = time.time()
        t_loss = train_one_epoch_transfer(model, train_loader, criterion, optimizer_s2, device, stage=2)
        v_loss, v_targets, v_preds = evaluate(model, val_loader, criterion, device)
        v_acc = accuracy_score(v_targets, v_preds)
        v_f1 = f1_score(v_targets, v_preds, average="macro", zero_division=0)
        ep_dur = time.time() - ep_t0

        if v_f1 > best_overall_macro_f1:
            best_overall_macro_f1 = v_f1
            best_overall_stage = 2
            best_overall_epoch = epoch
            train_loss_at_best = t_loss
            val_loss_at_best = v_loss
            patience_cnt_s2 = 0
            torch.save(model.state_dict(), run_ckpt_path)
            tag = f"[BEST Overall {v_f1:.4f}]"
        else:
            patience_cnt_s2 += 1
            tag = f"(patience {patience_cnt_s2}/5)"

        history.append({
            "stage": 2, "epoch": epoch,
            "train_loss": round(t_loss, 4), "val_loss": round(v_loss, 4),
            "val_acc": round(v_acc, 4), "val_f1": round(v_f1, 4),
        })

        cfg_label = "Aug" if use_augmentation else "NoAug"
        print(
            f"[{cfg_label}|S{seed}|S2-E{epoch:02d}/20] "
            f"TrainLoss: {t_loss:.4f} | ValLoss: {v_loss:.4f} | "
            f"ValAcc: {v_acc:.4f} | ValF1: {v_f1:.4f} | {ep_dur:.1f}s {tag}"
        )

        if patience_cnt_s2 >= 5:
            break

    total_time = time.time() - start_run_time
    if temp_s1_ckpt.exists():
        temp_s1_ckpt.unlink()

    # Final validation on best saved checkpoint
    model.load_state_dict(torch.load(run_ckpt_path, map_location=device))
    best_eval_loss, final_targets, final_preds = evaluate(model, val_loader, criterion, device)
    best_eval_acc = accuracy_score(final_targets, final_preds)
    best_eval_f1 = f1_score(final_targets, final_preds, average="macro", zero_division=0)

    clf_rep = classification_report(
        final_targets,
        final_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    final_train_loss = history[-1]["train_loss"]
    final_val_loss = history[-1]["val_loss"]

    run_record = {
        "config": config_name,
        "seed": seed,
        "val_accuracy": round(best_eval_acc, 4),
        "val_macro_f1": round(best_eval_f1, 4),
        "cardboard_f1": round(clf_rep["cardboard"]["f1-score"], 4),
        "glass_f1": round(clf_rep["glass"]["f1-score"], 4),
        "metal_f1": round(clf_rep["metal"]["f1-score"], 4),
        "paper_f1": round(clf_rep["paper"]["f1-score"], 4),
        "plastic_f1": round(clf_rep["plastic"]["f1-score"], 4),
        "trash_f1": round(clf_rep["trash"]["f1-score"], 4),
        "best_stage": best_overall_stage,
        "best_epoch": best_overall_epoch,
        "train_loss_at_best": round(train_loss_at_best, 4),
        "val_loss_at_best": round(val_loss_at_best, 4),
        "final_train_loss": round(final_train_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "loss_gap": round(val_loss_at_best - train_loss_at_best, 4),
        "training_time_sec": round(total_time, 2),
        "checkpoint_path": str(run_ckpt_path),
        "history": history,
    }
    return run_record


def get_phase4_seed42_record(reports_dir: Path, models_dir: Path) -> Dict:
    """Load exact metrics from Phase 4 for No-Augmentation seed 42."""
    with open(reports_dir / "transfer_val_metrics.json", "r") as f:
        p4_metrics = json.load(f)
    p4_hist = pd.read_csv(reports_dir / "transfer_history.csv")

    s2_best_row = p4_hist[(p4_hist["stage"] == 2) & (p4_hist["epoch"] == p4_metrics["best_epoch"])].iloc[0]
    final_row = p4_hist.iloc[-1]

    # Reconstruct history list
    history = []
    for _, row in p4_hist.iterrows():
        history.append({
            "stage": int(row["stage"]),
            "epoch": int(row["epoch"]),
            "train_loss": round(float(row["train_loss"]), 4),
            "val_loss": round(float(row["val_loss"]), 4),
            "val_acc": round(float(row["val_accuracy"]), 4),
            "val_f1": round(float(row["val_macro_f1"]), 4),
        })

    return {
        "config": "no_augmentation",
        "seed": 42,
        "val_accuracy": round(p4_metrics["val_accuracy"], 4),
        "val_macro_f1": round(p4_metrics["val_macro_f1"], 4),
        "cardboard_f1": round(p4_metrics["per_class"]["cardboard"]["f1_score"], 4),
        "glass_f1": round(p4_metrics["per_class"]["glass"]["f1_score"], 4),
        "metal_f1": round(p4_metrics["per_class"]["metal"]["f1_score"], 4),
        "paper_f1": round(p4_metrics["per_class"]["paper"]["f1_score"], 4),
        "plastic_f1": round(p4_metrics["per_class"]["plastic"]["f1_score"], 4),
        "trash_f1": round(p4_metrics["per_class"]["trash"]["f1_score"], 4),
        "best_stage": p4_metrics["best_stage"],
        "best_epoch": p4_metrics["best_epoch"],
        "train_loss_at_best": round(float(s2_best_row["train_loss"]), 4),
        "val_loss_at_best": round(float(s2_best_row["val_loss"]), 4),
        "final_train_loss": round(float(final_row["train_loss"]), 4),
        "final_val_loss": round(float(final_row["val_loss"]), 4),
        "loss_gap": round(float(s2_best_row["val_loss"] - s2_best_row["train_loss"]), 4),
        "training_time_sec": round(float(p4_metrics["total_training_time_seconds"]), 2),
        "checkpoint_path": str(models_dir / "mobilenetv2_transfer.pt"),
        "history": history,
    }


def main():
    print("=" * 75)
    print("EcoVision - Phase 5: Augmentation Comparison Experiment")
    print("=" * 75)

    if not torch.cuda.is_available():
        raise RuntimeError("Rule 5 & 7 Violation: CUDA GPU is required!")
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    # 1. Collect all 6 runs
    all_runs = []

    # Run 1: No Augmentation, Seed 42 (Reused from Phase 4)
    print("\n[REUSING] Phase 4 No-Augmentation Seed 42 baseline...")
    seed42_noaug = get_phase4_seed42_record(reports_dir, models_dir)
    all_runs.append(seed42_noaug)
    print(
        f"Reused No-Aug Seed 42: Val Acc={seed42_noaug['val_accuracy']*100:.2f}%, "
        f"Val Macro-F1={seed42_noaug['val_macro_f1']:.4f}, "
        f"Train/Val Gap={seed42_noaug['loss_gap']:.4f}"
    )

    # Remaining 5 runs
    configs_to_run = [
        ("no_augmentation", 43, False),
        ("no_augmentation", 44, False),
        ("augmentation", 42, True),
        ("augmentation", 43, True),
        ("augmentation", 44, True),
    ]

    for cfg_name, seed, use_aug in configs_to_run:
        print("\n" + "-" * 75)
        print(f"Starting Run: Config='{cfg_name}', Seed={seed}, Augmentation={use_aug}")
        print("-" * 75)
        record = run_experiment(
            config_name=cfg_name,
            seed=seed,
            use_augmentation=use_aug,
            data_dir=data_dir,
            models_dir=models_dir,
            device=device,
            class_names=class_names,
        )
        all_runs.append(record)
        print(
            f"Finished {cfg_name} Seed {seed}: Val Acc={record['val_accuracy']*100:.2f}%, "
            f"Val Macro-F1={record['val_macro_f1']:.4f}, "
            f"Time={record['training_time_sec']:.1f}s, Loss Gap={record['loss_gap']:.4f}"
        )

    # 2. Save reports/augmentation_runs.csv
    # Exclude internal non-csv objects like history
    csv_runs = []
    for r in all_runs:
        row = {k: v for k, v in r.items() if k not in ["history", "checkpoint_path"]}
        csv_runs.append(row)

    runs_df = pd.DataFrame(csv_runs)
    runs_path = reports_dir / "augmentation_runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"\n[SAVED] Per-run metrics saved to {runs_path}")

    # 3. Compute Summary Statistics (mean +- std over the 3 seeds)
    metrics_to_summarize = [
        ("val_accuracy", "Overall Accuracy (%)", 100.0),
        ("val_macro_f1", "Macro-F1 Score", 1.0),
        ("cardboard_f1", "Cardboard F1", 1.0),
        ("glass_f1", "Glass F1", 1.0),
        ("metal_f1", "Metal F1", 1.0),
        ("paper_f1", "Paper F1", 1.0),
        ("plastic_f1", "Plastic F1", 1.0),
        ("trash_f1", "Trash F1", 1.0),
        ("loss_gap", "Train/Val Loss Gap", 1.0),
        ("training_time_sec", "Training Time (s)", 1.0),
    ]

    summary_rows = []
    noaug_df = runs_df[runs_df["config"] == "no_augmentation"]
    aug_df = runs_df[runs_df["config"] == "augmentation"]

    for col, display_name, scale in metrics_to_summarize:
        noaug_mean = (noaug_df[col] * scale).mean()
        noaug_std = (noaug_df[col] * scale).std(ddof=1)
        aug_mean = (aug_df[col] * scale).mean()
        aug_std = (aug_df[col] * scale).std(ddof=1)
        delta_mean = aug_mean - noaug_mean

        if scale == 100.0:
            noaug_str = f"{noaug_mean:.2f}% ± {noaug_std:.2f}%"
            aug_str = f"{aug_mean:.2f}% ± {aug_std:.2f}%"
            delta_str = f"{delta_mean:+.2f}%"
        elif col in ["training_time_sec"]:
            noaug_str = f"{noaug_mean:.1f}s ± {noaug_std:.1f}s"
            aug_str = f"{aug_mean:.1f}s ± {aug_std:.1f}s"
            delta_str = f"{delta_mean:+.1f}s"
        else:
            noaug_str = f"{noaug_mean:.4f} ± {noaug_std:.4f}"
            aug_str = f"{aug_mean:.4f} ± {aug_std:.4f}"
            delta_str = f"{delta_mean:+.4f}"

        summary_rows.append({
            "metric": display_name,
            "no_augmentation": noaug_str,
            "augmentation": aug_str,
            "delta (aug - noaug)": delta_str,
            "noaug_mean_raw": noaug_mean,
            "noaug_std_raw": noaug_std,
            "aug_mean_raw": aug_mean,
            "aug_std_raw": aug_std,
            "delta_mean_raw": delta_mean,
        })

    summary_df = pd.DataFrame(summary_rows)
    # Save clean user-facing columns to csv
    clean_summary_df = summary_df[["metric", "no_augmentation", "augmentation", "delta (aug - noaug)"]]
    summary_path = reports_dir / "augmentation_summary.csv"
    clean_summary_df.to_csv(summary_path, index=False)
    print(f"[SAVED] Summary table saved to {summary_path}")

    # 4. Generate Visualizations
    # 4a. reports/augmentation_effect.png: Bar chart with error bars
    classes_and_macro = [
        ("Macro-F1 Score", "val_macro_f1"),
        ("Overall Accuracy", "val_accuracy"),
        ("Cardboard F1", "cardboard_f1"),
        ("Glass F1", "glass_f1"),
        ("Metal F1", "metal_f1"),
        ("Paper F1", "paper_f1"),
        ("Plastic F1", "plastic_f1"),
        ("Trash F1", "trash_f1"),
    ]

    labels = [item[0] for item in classes_and_macro]
    x = np.arange(len(labels))
    width = 0.35

    noaug_means = [noaug_df[item[1]].mean() * 100.0 for item in classes_and_macro]
    noaug_stds = [noaug_df[item[1]].std(ddof=1) * 100.0 for item in classes_and_macro]

    aug_means = [aug_df[item[1]].mean() * 100.0 for item in classes_and_macro]
    aug_stds = [aug_df[item[1]].std(ddof=1) * 100.0 for item in classes_and_macro]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, noaug_means, width, yerr=noaug_stds, label="No Augmentation", capsize=5, color="#4C72B0", alpha=0.9, edgecolor="black")
    bars2 = ax.bar(x + width/2, aug_means, width, yerr=aug_stds, label="Augmentation", capsize=5, color="#55A868", alpha=0.9, edgecolor="black")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("EcoVision Phase 5: Augmentation Effect on MobileNetV2 Validation Metrics\n(Mean ± 1 Std Dev across Seeds 42, 43, 44)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=True, fontsize=11)

    # Bar value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", color="#1f2d3d", fontweight="bold", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1.5, f"{h:.1f}%", ha="center", va="bottom", color="#1e3d2f", fontweight="bold", fontsize=8)

    plt.tight_layout()
    effect_plot_path = reports_dir / "augmentation_effect.png"
    plt.savefig(effect_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Augmentation effect plot saved to {effect_plot_path}")

    # 4b. reports/augmentation_gap.png: Train/val loss gap comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Final Train Loss vs Final Val Loss by Config
    loss_comparison_data = []
    for r in all_runs:
        loss_comparison_data.append({"Config": "No Augmentation" if r["config"] == "no_augmentation" else "Augmentation", "Split": "Train Loss (Best)", "Loss": r["train_loss_at_best"]})
        loss_comparison_data.append({"Config": "No Augmentation" if r["config"] == "no_augmentation" else "Augmentation", "Split": "Val Loss (Best)", "Loss": r["val_loss_at_best"]})
    loss_df = pd.DataFrame(loss_comparison_data)

    sns.barplot(data=loss_df, x="Config", y="Loss", hue="Split", ax=ax1, palette=["#34495e", "#e74c3c"], capsize=0.1, edgecolor="black")
    ax1.set_title("CrossEntropy Loss at Best Checkpoint Epoch\n(Mean ± Std across Seeds)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Generalization Gap (Val Loss - Train Loss)
    gap_data = [
        {"Config": "No Augmentation" if r["config"] == "no_augmentation" else "Augmentation", "Loss Gap (Val - Train)": r["loss_gap"], "Seed": r["seed"]}
        for r in all_runs
    ]
    gap_df = pd.DataFrame(gap_data)

    sns.barplot(data=gap_df, x="Config", y="Loss Gap (Val - Train)", ax=ax2, palette=["#4C72B0", "#55A868"], capsize=0.1, edgecolor="black")
    sns.stripplot(data=gap_df, x="Config", y="Loss Gap (Val - Train)", ax=ax2, color="black", size=7, jitter=0.1)
    ax2.set_title("Generalization Loss Gap (Val Loss - Train Loss)\n(Lower Gap = Reduced Overfitting)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Loss Gap", fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("EcoVision Phase 5: Generalization & Overfitting Analysis", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    gap_plot_path = reports_dir / "augmentation_gap.png"
    plt.savefig(gap_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Loss gap analysis plot saved to {gap_plot_path}")

    # 5. Selection Rule Evaluation & Final Model Checkpoint
    # Rule: Pick final configuration by mean validation macro-F1.
    # If difference is within one std dev, state not a clear difference and prefer simpler config
    # unless augmentation reduces the train/val gap.
    macro_row = summary_df[summary_df["metric"] == "Macro-F1 Score"].iloc[0]
    gap_row = summary_df[summary_df["metric"] == "Train/Val Loss Gap"].iloc[0]

    noaug_f1_mean = macro_row["noaug_mean_raw"]
    noaug_f1_std = macro_row["noaug_std_raw"]
    aug_f1_mean = macro_row["aug_mean_raw"]
    aug_f1_std = macro_row["aug_std_raw"]
    f1_diff = aug_f1_mean - noaug_f1_mean

    pooled_std = np.sqrt((noaug_f1_std**2 + aug_f1_std**2) / 2)
    is_diff_within_1std = abs(f1_diff) <= max(noaug_f1_std, aug_f1_std, pooled_std)

    noaug_gap_mean = gap_row["noaug_mean_raw"]
    aug_gap_mean = gap_row["aug_mean_raw"]
    aug_reduces_gap = aug_gap_mean < noaug_gap_mean

    print("\n" + "=" * 75)
    print("DECISION ANALYSIS: Configuration Selection Rule")
    print("=" * 75)
    print(f"No Augmentation Mean Macro-F1: {noaug_f1_mean:.4f} ± {noaug_f1_std:.4f}")
    print(f"Augmentation    Mean Macro-F1: {aug_f1_mean:.4f} ± {aug_f1_std:.4f}")
    print(f"Macro-F1 Delta (Aug - NoAug) : {f1_diff:+.4f} (Pooled Std: {pooled_std:.4f})")
    print(f"Within 1 Std Dev Threshold   : {is_diff_within_1std}")
    print(f"Train/Val Loss Gap: NoAug={noaug_gap_mean:.4f} vs Aug={aug_gap_mean:.4f} (Aug reduces gap: {aug_reduces_gap})")

    if not is_diff_within_1std:
        chosen_config = "augmentation" if f1_diff > 0 else "no_augmentation"
        decision_reason = (
            f"Statistically significant difference outside 1 standard deviation: "
            f"{chosen_config} outperforms by {abs(f1_diff):.4f} in validation macro-F1."
        )
    else:
        if aug_reduces_gap and f1_diff >= 0:
            chosen_config = "augmentation"
            decision_reason = (
                f"Validation macro-F1 difference ({f1_diff:+.4f}) is within 1 standard deviation ({pooled_std:.4f}), "
                f"meaning macro-F1 difference is not statistically distinct. However, augmentation significantly "
                f"reduces the train/val generalization gap ({aug_gap_mean:.4f} vs {noaug_gap_mean:.4f}), mitigating overfitting."
            )
        elif aug_reduces_gap and f1_diff < 0 and abs(f1_diff) < 0.005:
            chosen_config = "augmentation"
            decision_reason = (
                f"Validation macro-F1 difference is negligible ({f1_diff:+.4f}) and within 1 std dev, "
                f"while augmentation successfully reduces the generalization gap ({aug_gap_mean:.4f} vs {noaug_gap_mean:.4f})."
            )
        else:
            chosen_config = "no_augmentation"
            decision_reason = (
                f"Validation macro-F1 difference ({f1_diff:+.4f}) is within 1 standard deviation ({pooled_std:.4f}), "
                f"indicating no clear statistical advantage. Per Occam's razor / engineering guidelines, "
                f"preferring the simpler un-augmented pipeline."
            )

    print(f"\n>>> Chosen Configuration: '{chosen_config.upper()}'")
    print(f">>> Rationale: {decision_reason}")

    # Find best seed of chosen config
    chosen_runs = [r for r in all_runs if r["config"] == chosen_config]
    best_chosen_run = max(chosen_runs, key=lambda x: x["val_macro_f1"])
    best_seed = best_chosen_run["seed"]
    best_ckpt_source = Path(best_chosen_run["checkpoint_path"])

    final_model_dest = models_dir / "mobilenetv2_final.pt"
    shutil.copyfile(best_ckpt_source, final_model_dest)
    print(f"\n[SAVED] Copied best model ({chosen_config} Seed {best_seed}, Val Macro-F1={best_chosen_run['val_macro_f1']:.4f}) to {final_model_dest}")

    # 6. Print Console Summary Table
    print("\n" + "=" * 75)
    print("PHASE 5 SUMMARY: No-Augmentation vs Augmentation (N=3 Seeds Each)")
    print("=" * 75)
    print(f"{'Metric':<25} | {'No Augmentation':<18} | {'Augmentation':<18} | {'Delta':<10}")
    print("-" * 75)
    for _, row in clean_summary_df.iterrows():
        print(f"{row['metric']:<25} | {row['no_augmentation']:<18} | {row['augmentation']:<18} | {row['delta (aug - noaug)']:<10}")
    print("=" * 75)


if __name__ == "__main__":
    main()
