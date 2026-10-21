"""EcoVision Phase 2: Deduplication and Stratified Group Splitting.

1. Restricts modeling data to the unique 2,527 resized (512x384) TrashNet images.
2. Removes the 3 conflicting exact-duplicate pairs (6 images total) by filename
   and logs them to reports/removed_images.csv.
3. Analyzes near-duplicate Hamming distance thresholds (0, 2, 4, 6), justifies
   threshold <= 4, and visualizes representative pairs in reports/near_duplicates_examples.png.
4. Performs a deterministic Stratified GROUP Split (70/15/15, seed 42) ensuring no
   group or near-duplicate bridges across splits.
5. Resizes and caches 224x224 images into data/processed/ (train/val/test).
"""

import json
from collections import defaultdict
from pathlib import Path
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

RANDOM_SEED = 42

# 3 pairs of exact duplicates identified with conflicting labels
CONFLICTING_FILENAMES = [
    "glass115.jpg", "metal91.jpg",
    "glass176.jpg", "plastic152.jpg",
    "glass389.jpg", "plastic332.jpg",
]

TARGET_IMG_SIZE = (224, 224)


def run_dedupe_and_split():
    print("=" * 60)
    print("EcoVision - Phase 2: Deduplication & Stratified Group Split")
    print("=" * 60)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata for resized archive only (archive == 1)
    df_meta = pd.read_csv(reports_dir / "dataset_metadata.csv")
    df_resized = df_meta[df_meta["archive"] == 1].copy().reset_index(drop=True)
    initial_count = len(df_resized)
    print(f"[1/5] Filtered to resized archive: {initial_count} unique images.")

    # 2. Excise conflicting exact duplicates
    removed_df = df_resized[df_resized["filename"].isin(CONFLICTING_FILENAMES)].copy()
    conflict_pairs = {
        "glass115.jpg": ("metal91.jpg", "glass vs metal contradiction"),
        "metal91.jpg": ("glass115.jpg", "glass vs metal contradiction"),
        "glass176.jpg": ("plastic152.jpg", "glass vs plastic contradiction"),
        "plastic152.jpg": ("glass176.jpg", "glass vs plastic contradiction"),
        "glass389.jpg": ("plastic332.jpg", "glass vs plastic contradiction"),
        "plastic332.jpg": ("glass389.jpg", "glass vs plastic contradiction"),
    }
    removed_df["conflicting_pair"] = removed_df["filename"].map(lambda f: conflict_pairs[f][0])
    removed_df["removal_reason"] = removed_df["filename"].map(lambda f: conflict_pairs[f][1])

    removed_csv_path = reports_dir / "removed_images.csv"
    removed_df[["filename", "label_name", "md5", "phash", "conflicting_pair", "removal_reason"]].to_csv(
        removed_csv_path, index=False
    )
    print(f"[2/5] Removed {len(removed_df)} conflicting images. Logged to {removed_csv_path}.")

    df_clean = df_resized[~df_resized["filename"].isin(CONFLICTING_FILENAMES)].copy().reset_index(drop=True)
    clean_count = len(df_clean)
    print(f"      Clean benchmark size: {clean_count} images.")

    # 3. Near-Duplicate Threshold Evidence (0, 2, 4, 6)
    print("[3/5] Evaluating perceptual hash Hamming distance thresholds (0, 2, 4, 6)...")
    phashes = [imagehash.hex_to_hash(h) for h in df_clean["phash"]]
    N = clean_count

    threshold_results = {}
    chosen_threshold = 4
    pairs_by_dist = defaultdict(list)

    for thresh in [0, 2, 4, 6]:
        parent = list(range(N))

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        pair_count = 0
        cross_class = 0
        for i in range(N):
            for j in range(i + 1, N):
                d = phashes[i] - phashes[j]
                if d <= thresh:
                    union(i, j)
                    pair_count += 1
                    if df_clean.iloc[i]["label_name"] != df_clean.iloc[j]["label_name"]:
                        cross_class += 1
                    if thresh == chosen_threshold:
                        pairs_by_dist[d].append((i, j))

        groups = defaultdict(list)
        for i in range(N):
            groups[find(i)].append(i)

        dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
        max_group_size = max(len(v) for v in groups.values()) if groups else 1
        total_in_dup_groups = sum(len(v) for v in dup_groups.values())

        threshold_results[f"threshold_{thresh}"] = {
            "hamming_threshold": thresh,
            "pairs_within_threshold": pair_count,
            "cross_class_pairs": cross_class,
            "duplicate_groups_count": len(dup_groups),
            "total_images_in_groups": total_in_dup_groups,
            "largest_group_size": max_group_size,
        }
        print(
            f"      Threshold <= {thresh}: {len(dup_groups)} duplicate groups | "
            f"{total_in_dup_groups} images | largest group: {max_group_size} | cross-class: {cross_class}"
        )

    # Save metrics JSON
    with open(reports_dir / "near_duplicates_metrics.json", "w", encoding="utf-8") as f:
        json.dump(threshold_results, f, indent=2)

    # Build connected components for chosen threshold (<= 4)
    parent = list(range(N))

    def find_chosen(i):
        if parent[i] == i:
            return i
        parent[i] = find_chosen(parent[i])
        return parent[i]

    def union_chosen(i, j):
        root_i, root_j = find_chosen(i), find_chosen(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(N):
        for j in range(i + 1, N):
            if phashes[i] - phashes[j] <= chosen_threshold:
                union_chosen(i, j)

    df_clean["group_id"] = [find_chosen(i) for i in range(N)]

    # Save example near-duplicate pairs image grid (6 pairs)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    sample_pair_indices = []
    for d in sorted(pairs_by_dist.keys()):
        for p in pairs_by_dist[d]:
            sample_pair_indices.append((d, p[0], p[1]))
            if len(sample_pair_indices) >= 6:
                break
        if len(sample_pair_indices) >= 6:
            break

    for row_idx, (d, idx1, idx2) in enumerate(sample_pair_indices[:3]):
        r1, r2 = df_clean.iloc[idx1], df_clean.iloc[idx2]
        img1 = Image.open(r1["image_path"])
        img2 = Image.open(r2["image_path"])

        axes[row_idx, 0].imshow(img1)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title(f"{r1['label_name']}: {r1['filename']}", fontsize=9, fontweight="bold")

        axes[row_idx, 1].imshow(img2)
        axes[row_idx, 1].axis("off")
        axes[row_idx, 1].set_title(f"{r2['label_name']}: {r2['filename']}\n(Hamming Dist = {d})", fontsize=9, fontweight="bold")

    for row_idx, (d, idx1, idx2) in enumerate(sample_pair_indices[3:6]):
        r1, r2 = df_clean.iloc[idx1], df_clean.iloc[idx2]
        img1 = Image.open(r1["image_path"])
        img2 = Image.open(r2["image_path"])

        axes[row_idx, 2].imshow(img1)
        axes[row_idx, 2].axis("off")
        axes[row_idx, 2].set_title(f"{r1['label_name']}: {r1['filename']}", fontsize=9, fontweight="bold")

        axes[row_idx, 3].imshow(img2)
        axes[row_idx, 3].axis("off")
        axes[row_idx, 3].set_title(f"{r2['label_name']}: {r2['filename']}\n(Hamming Dist = {d})", fontsize=9, fontweight="bold")

    plt.suptitle("Example Near-Duplicate Pairs Identified via Perceptual Hash (Threshold <= 4)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(reports_dir / "near_duplicates_examples.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] Near duplicate visual grid saved to {reports_dir / 'near_duplicates_examples.png'}")

    # 4. Stratified GROUP Split (70 / 15 / 15, seed 42)
    print("[4/5] Executing deterministic Stratified Group Split (70/15/15, seed 42)...")
    groups_by_label = defaultdict(list)
    for gid, group_df in df_clean.groupby("group_id"):
        primary_label = group_df["label_name"].value_counts().index[0]
        groups_by_label[primary_label].append((gid, group_df))

    rng = np.random.RandomState(RANDOM_SEED)
    train_gids, val_gids, test_gids = set(), set(), set()

    for lbl in sorted(groups_by_label.keys()):
        g_list = groups_by_label[lbl]
        indices = list(range(len(g_list)))
        rng.shuffle(indices)
        shuffled_groups = [g_list[i] for i in indices]

        total_samples = sum(len(gdf) for _, gdf in shuffled_groups)
        target_val = int(round(total_samples * 0.15))
        target_test = int(round(total_samples * 0.15))

        val_count, test_count = 0, 0
        for gid, gdf in shuffled_groups:
            c = len(gdf)
            if val_count < target_val:
                val_gids.add(gid)
                val_count += c
            elif test_count < target_test:
                test_gids.add(gid)
                test_count += c
            else:
                train_gids.add(gid)

    df_clean["split"] = "none"
    df_clean.loc[df_clean["group_id"].isin(train_gids), "split"] = "train"
    df_clean.loc[df_clean["group_id"].isin(val_gids), "split"] = "val"
    df_clean.loc[df_clean["group_id"].isin(test_gids), "split"] = "test"

    # Strict Real Verification Assertions
    split_map = dict(zip(df_clean["filename"], df_clean["split"]))

    # 1. Pairwise check: EVERY pair with pHash Hamming distance <= chosen_threshold must share the same split
    leakage_pairs = []
    for d in range(chosen_threshold + 1):
        for i, j in pairs_by_dist[d]:
            f1, f2 = df_clean.iloc[i]["filename"], df_clean.iloc[j]["filename"]
            s1, s2 = split_map[f1], split_map[f2]
            if s1 != s2:
                leakage_pairs.append((d, f1, s1, f2, s2))
    assert len(leakage_pairs) == 0, f"Perceptual hash leakage found across splits: {leakage_pairs}"

    # 2. Filename uniqueness across splits
    train_files = set(df_clean[df_clean["split"] == "train"]["filename"])
    val_files = set(df_clean[df_clean["split"] == "val"]["filename"])
    test_files = set(df_clean[df_clean["split"] == "test"]["filename"])
    assert len(train_files.intersection(val_files)) == 0, "Filename overlap between train and val!"
    assert len(train_files.intersection(test_files)) == 0, "Filename overlap between train and test!"
    assert len(val_files.intersection(test_files)) == 0, "Filename overlap between val and test!"

    # 3. MD5 uniqueness across splits
    train_md5s = set(df_clean[df_clean["split"] == "train"]["md5"])
    val_md5s = set(df_clean[df_clean["split"] == "val"]["md5"])
    test_md5s = set(df_clean[df_clean["split"] == "test"]["md5"])
    assert len(train_md5s.intersection(val_md5s)) == 0, "MD5 hash overlap between train and val!"
    assert len(train_md5s.intersection(test_md5s)) == 0, "MD5 hash overlap between train and test!"
    assert len(val_md5s.intersection(test_md5s)) == 0, "MD5 hash overlap between val and test!"

    # 4. Total counts and no unassigned images
    assert (df_clean["split"] != "none").all(), "Unassigned samples found!"
    assert len(df_clean) == 2521, f"Expected 2,521 clean images, got {len(df_clean)}"

    crosstab = pd.crosstab(df_clean["label_name"], df_clean["split"])
    assert (crosstab > 0).all().all(), "Every class must be present in every split!"
    print("      Verification passed: All pairwise pHash (<=4), filename, and MD5 leak checks succeeded.")

    # Save split CSVs
    train_df = df_clean[df_clean["split"] == "train"].copy()
    val_df = df_clean[df_clean["split"] == "val"].copy()
    test_df = df_clean[df_clean["split"] == "test"].copy()

    train_df.to_csv(reports_dir / "train_split.csv", index=False)
    val_df.to_csv(reports_dir / "val_split.csv", index=False)
    test_df.to_csv(reports_dir / "test_split.csv", index=False)

    summary_df = crosstab.copy()
    summary_df["total"] = summary_df.sum(axis=1)
    summary_df.loc["TOTAL"] = summary_df.sum(axis=0)
    summary_df.to_csv(reports_dir / "split_summary.csv")
    print(f"[SAVED] Split CSVs saved to {reports_dir}/(train_split, val_split, test_split, split_summary).csv")

    # 5. Resize and Cache 224x224 Images
    print("[5/5] Resizing and caching 224x224 images into data/processed/...")
    processed_dir = Path("data/processed")
    for split_name in ["train", "val", "test"]:
        for c in df_clean["label_name"].unique():
            (processed_dir / split_name / c).mkdir(parents=True, exist_ok=True)

    for idx, row in df_clean.iterrows():
        src_path = Path(row["image_path"])
        dst_path = processed_dir / row["split"] / row["label_name"] / row["filename"]
        if not dst_path.exists():
            img = Image.open(src_path).convert("RGB")
            img_resized = img.resize(TARGET_IMG_SIZE, Image.Resampling.BILINEAR)
            img_resized.save(dst_path, quality=95)

        if (idx + 1) % 500 == 0 or idx == len(df_clean) - 1:
            print(f"      Cached {idx + 1}/{len(df_clean)} images to 224x224...", flush=True)

    print("=" * 60)
    print("EcoVision Phase 2 Split Summary:")
    print("=" * 60)
    print(summary_df.to_string())
    print("=" * 60)
    print("[SUCCESS] Phase 2 Deduplication & Splitting Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_dedupe_and_split()
