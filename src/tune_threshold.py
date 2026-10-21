"""Phase 7: Confidence Threshold Tuning for EcoVision.

Tuning is conducted STRICTLY on reports/val_predictions.csv.
The selected threshold is then evaluated once on reports/test_predictions.csv.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def main():
    print("=" * 75)
    print("EcoVision - Phase 7: Confidence Threshold Optimization")
    print("=" * 75)

    reports_dir = project_root / "reports"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    val_pred_path = reports_dir / "val_predictions.csv"
    test_pred_path = reports_dir / "test_predictions.csv"

    if not val_pred_path.exists():
        raise FileNotFoundError(f"Missing validation predictions: {val_pred_path}")
    if not test_pred_path.exists():
        raise FileNotFoundError(f"Missing test predictions: {test_pred_path}")

    val_df = pd.read_csv(val_pred_path)
    print(f"Loaded validation predictions: {len(val_df)} rows")

    # -------------------------------------------------------------
    # 1. Sweep Thresholds on Validation Set ONLY (0.30 to 0.95 step 0.05)
    # -------------------------------------------------------------
    thresholds = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]
    sweep_records = []
    total_val = len(val_df)

    for t in thresholds:
        answered_mask = val_df["confidence"] >= t
        num_answered = int(answered_mask.sum())
        num_rejected = total_val - num_answered
        coverage = num_answered / total_val

        if num_answered > 0:
            sub_true = val_df.loc[answered_mask, "true"]
            sub_pred = val_df.loc[answered_mask, "pred"]
            acc = accuracy_score(sub_true, sub_pred)
            macro_f1 = f1_score(sub_true, sub_pred, average="macro", zero_division=0)
        else:
            acc = 0.0
            macro_f1 = 0.0

        sweep_records.append({
            "threshold": t,
            "coverage": round(coverage, 4),
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "num_answered": num_answered,
            "num_rejected": num_rejected,
            "acc_x_coverage": round(acc * coverage, 4),
        })

    sweep_df = pd.DataFrame(sweep_records)
    sweep_csv_path = reports_dir / "threshold_sweep.csv"
    # User-facing columns
    export_df = sweep_df[["threshold", "coverage", "accuracy", "macro_f1", "num_answered", "num_rejected"]]
    export_df.to_csv(sweep_csv_path, index=False)
    print(f"[SAVED] Threshold sweep table saved to {sweep_csv_path}")

    # -------------------------------------------------------------
    # 2. Selection Rule (Fixed in Advance)
    # "Pick threshold with highest coverage such that validation accuracy >= 0.95.
    #  If none qualifies, pick the one maximizing accuracy x coverage and say so."
    # -------------------------------------------------------------
    qualifying = sweep_df[sweep_df["accuracy"] >= 0.95]

    if not qualifying.empty:
        # Highest coverage means lowest threshold among qualifying rows
        chosen_row = qualifying.sort_values(by=["coverage", "threshold"], ascending=[False, True]).iloc[0]
        rule_desc = "Highest coverage with validation accuracy >= 0.95"
    else:
        chosen_row = sweep_df.sort_values(by="acc_x_coverage", ascending=False).iloc[0]
        rule_desc = "No threshold achieved validation accuracy >= 0.95; maximized accuracy * coverage"

    chosen_threshold = float(chosen_row["threshold"])
    print(f"\nSelection Rule: {rule_desc}")
    print(f"Chosen Validation Threshold: {chosen_threshold:.2f}")
    print(
        f"Validation at Threshold: Coverage={chosen_row['coverage']*100:.2f}%, "
        f"Accuracy={chosen_row['accuracy']*100:.2f}%, "
        f"Macro-F1={chosen_row['macro_f1']:.4f}, "
        f"Answered={int(chosen_row['num_answered'])}/{total_val}, "
        f"Rejected={int(chosen_row['num_rejected'])}"
    )

    threshold_payload = {
        "chosen_threshold": chosen_threshold,
        "selection_rule": rule_desc,
        "validation": {
            "threshold": chosen_threshold,
            "coverage": float(chosen_row["coverage"]),
            "accuracy": float(chosen_row["accuracy"]),
            "macro_f1": float(chosen_row["macro_f1"]),
            "num_answered": int(chosen_row["num_answered"]),
            "num_rejected": int(chosen_row["num_rejected"]),
            "total_samples": total_val,
        },
    }
    threshold_json_path = models_dir / "threshold.json"
    with open(threshold_json_path, "w") as f:
        json.dump(threshold_payload, f, indent=2)
    print(f"[SAVED] Chosen threshold saved to {threshold_json_path}")

    # -------------------------------------------------------------
    # 3. Plot Threshold Curve
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Accuracy & Macro-F1 vs Threshold
    ax1.plot(sweep_df["threshold"], sweep_df["accuracy"] * 100, "o-", color="#2980b9", label="Accuracy (%)", linewidth=2)
    ax1.plot(sweep_df["threshold"], sweep_df["macro_f1"] * 100, "s--", color="#27ae60", label="Macro-F1 (%)", linewidth=2)
    ax1.axvline(chosen_threshold, color="#e74c3c", linestyle=":", linewidth=2, label=f"Selected Threshold ({chosen_threshold:.2f})")
    ax1.axhline(95.0, color="#8e44ad", linestyle="--", alpha=0.7, label="95% Target Accuracy")
    ax1.set_xlabel("Confidence Threshold", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Validation Accuracy & Macro-F1 vs. Threshold", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    # Panel 2: Coverage vs Threshold & Accuracy vs Coverage Trade-off
    ax2.plot(sweep_df["threshold"], sweep_df["coverage"] * 100, "d-", color="#e67e22", label="Coverage (%)", linewidth=2)
    ax2.axvline(chosen_threshold, color="#e74c3c", linestyle=":", linewidth=2, label=f"Selected Threshold ({chosen_threshold:.2f})")
    ax2.scatter(
        [chosen_threshold],
        [chosen_row["coverage"] * 100],
        color="#e74c3c",
        s=120,
        zorder=5,
        label=f"Selected Point ({chosen_row['coverage']*100:.1f}% cov)",
    )
    ax2.set_xlabel("Confidence Threshold", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Coverage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Coverage vs. Threshold", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.suptitle("EcoVision Phase 7: Confidence Threshold Tuning (Validation Set)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    curve_plot_path = reports_dir / "threshold_curve.png"
    plt.savefig(curve_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Threshold curve plot saved to {curve_plot_path}")

    # -------------------------------------------------------------
    # 4. Apply Chosen Threshold ONCE to Test Set
    # -------------------------------------------------------------
    test_df = pd.read_csv(test_pred_path)
    total_test = len(test_df)
    test_answered_mask = test_df["confidence"] >= chosen_threshold
    num_test_answered = int(test_answered_mask.sum())
    num_test_rejected = total_test - num_test_answered
    test_coverage = num_test_answered / total_test

    if num_test_answered > 0:
        test_sub_true = test_df.loc[test_answered_mask, "true"]
        test_sub_pred = test_df.loc[test_answered_mask, "pred"]
        test_acc = accuracy_score(test_sub_true, test_sub_pred)
        test_macro_f1 = f1_score(test_sub_true, test_sub_pred, average="macro", zero_division=0)
    else:
        test_acc = 0.0
        test_macro_f1 = 0.0

    test_threshold_payload = {
        "chosen_threshold": chosen_threshold,
        "selection_rule": rule_desc,
        "test": {
            "threshold": chosen_threshold,
            "coverage": round(float(test_coverage), 4),
            "accuracy": round(float(test_acc), 4),
            "macro_f1": round(float(test_macro_f1), 4),
            "num_answered": num_test_answered,
            "num_rejected": num_test_rejected,
            "total_samples": total_test,
        },
        "raw_test_baseline": {
            "accuracy": round(float(accuracy_score(test_df["true"], test_df["pred"])), 4),
            "macro_f1": round(float(f1_score(test_df["true"], test_df["pred"], average="macro", zero_division=0)), 4),
        },
    }

    test_threshold_path = reports_dir / "test_with_threshold.json"
    with open(test_threshold_path, "w") as f:
        json.dump(test_threshold_payload, f, indent=2)
    print(f"[SAVED] Test results with threshold saved to {test_threshold_path}")

    # -------------------------------------------------------------
    # 5. CHECK 2 Verifications
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("CHECK 2: Confidence Threshold Verification")
    print("=" * 75)
    print("1. Confirmation: Threshold was chosen STRICTLY from validation set predictions.")
    print(f"2. Chosen Threshold Row in Sweep Table:")
    print(f"   Threshold    : {chosen_row['threshold']:.2f}")
    print(f"   Val Coverage : {chosen_row['coverage']*100:.2f}% ({int(chosen_row['num_answered'])}/{total_val} images)")
    print(f"   Val Accuracy : {chosen_row['accuracy']*100:.2f}% (Target: >= 95.0%)")
    print(f"   Val Macro-F1 : {chosen_row['macro_f1']:.4f}")
    print(f"   Val Rejected : {int(chosen_row['num_rejected'])}")
    print(f"\n3. Holdout Test Set Evaluation with Chosen Threshold ({chosen_threshold:.2f}):")
    print(f"   Test Coverage: {test_coverage*100:.2f}% ({num_test_answered}/{total_test} images)")
    print(f"   Test Accuracy: {test_acc*100:.2f}% (Raw Test Accuracy: {test_threshold_payload['raw_test_baseline']['accuracy']*100:.2f}%)")
    print(f"   Accuracy Gain: {(test_acc - test_threshold_payload['raw_test_baseline']['accuracy'])*100:+.2f}% on answered samples")
    print(f"   Test Macro-F1: {test_macro_f1:.4f} (Raw Test Macro-F1: {test_threshold_payload['raw_test_baseline']['macro_f1']:.4f})")
    print(f"   Test Rejected: {num_test_rejected} ({num_test_rejected/total_test*100:.2f}%)")
    print("=" * 75)


if __name__ == "__main__":
    main()
