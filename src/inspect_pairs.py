"""Inspect and visualize cross-class pairs and borderline near-duplicate pairs."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import imagehash


def main():
    reports_dir = Path("reports")
    df = pd.read_csv(reports_dir / "dataset_metadata.csv")
    df_resized = df[df["archive"] == 1].copy().reset_index(drop=True)
    conflicting = [
        "glass115.jpg", "metal91.jpg",
        "glass176.jpg", "plastic152.jpg",
        "glass389.jpg", "plastic332.jpg",
    ]
    df_clean = df_resized[~df_resized["filename"].isin(conflicting)].copy().reset_index(drop=True)

    phashes = [imagehash.hex_to_hash(h) for h in df_clean["phash"]]
    N = len(df_clean)

    # 1. Identify Cross-Class Pairs at threshold <= 4
    cross_class_pairs = []
    borderline_pairs = []  # distance 3 and 4

    for i in range(N):
        for j in range(i + 1, N):
            d = phashes[i] - phashes[j]
            if d <= 4:
                r1 = df_clean.iloc[i]
                r2 = df_clean.iloc[j]
                if r1["label_name"] != r2["label_name"]:
                    cross_class_pairs.append((d, r1, r2))
                if d in (3, 4):
                    borderline_pairs.append((d, r1, r2))

    print(f"Found {len(cross_class_pairs)} cross-class pairs at distance <= 4:")
    for d, r1, r2 in cross_class_pairs:
        print(f"  Dist={d}: {r1['label_name']}/{r1['filename']} vs {r2['label_name']}/{r2['filename']}")

    # Save Cross-Class Pairs Grid
    fig, axes = plt.subplots(len(cross_class_pairs), 2, figsize=(10, 4 * len(cross_class_pairs)))
    if len(cross_class_pairs) == 1:
        axes = axes.reshape(1, 2)

    for i, (d, r1, r2) in enumerate(cross_class_pairs):
        img1 = Image.open(r1["image_path"])
        img2 = Image.open(r2["image_path"])

        axes[i, 0].imshow(img1)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"{r1['label_name'].upper()}: {r1['filename']}", fontsize=11, fontweight="bold")

        axes[i, 1].imshow(img2)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"{r2['label_name'].upper()}: {r2['filename']}\n(Hamming Distance = {d})", fontsize=11, fontweight="bold")

    plt.suptitle("Cross-Class Near-Duplicate Pairs at Hamming Distance <= 4", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    cross_path = reports_dir / "cross_class_pairs.png"
    plt.savefig(cross_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Cross-class pairs grid saved to {cross_path}")

    # 2. Save Borderline Near-Duplicates Grid (Distances 3 and 4, up to 6 pairs)
    borderline_pairs.sort(key=lambda x: (x[0], x[1]["filename"]))
    selected_borderline = borderline_pairs[:6]
    print(f"Total borderline pairs (dist 3 and 4): {len(borderline_pairs)}, displaying {len(selected_borderline)}")

    num_pairs = len(selected_borderline)
    fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 3.5 * num_pairs))
    if num_pairs == 1:
        axes = axes.reshape(1, 2)

    for i, (d, r1, r2) in enumerate(selected_borderline):
        img1 = Image.open(r1["image_path"])
        img2 = Image.open(r2["image_path"])

        axes[i, 0].imshow(img1)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"{r1['label_name'].upper()}: {r1['filename']}", fontsize=10, fontweight="bold")

        axes[i, 1].imshow(img2)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"{r2['label_name'].upper()}: {r2['filename']} (Hamming Dist = {d})", fontsize=10, fontweight="bold")

    plt.suptitle("Borderline Near-Duplicate Pairs (Hamming Distance 3 & 4)", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    borderline_path = reports_dir / "near_duplicates_borderline.png"
    plt.savefig(borderline_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Borderline near duplicates grid saved to {borderline_path}")


if __name__ == "__main__":
    main()
