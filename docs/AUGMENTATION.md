# Data Augmentation Experiment Report (AUGMENTATION.md)

## 1. Overview & Experimental Design

Phase 5 evaluates whether applying stochastic data augmentation during training improves the generalization and robustness of the fine-tuned **MobileNetV2** model on the TrashNet dataset.

To ensure scientific rigor and prevent false conclusions from random initialization:
- **Seed Replication:** Both configurations are evaluated across **3 random seeds (42, 43, 44)**.
- **Strict Split Isolation:** All evaluations are conducted strictly on the **validation split (379 images)**. The **test set remains completely untouched**.
- **Consistent Optimization Pipeline:** Both configurations use the identical two-stage transfer learning recipe established in Phase 4:
  - **Stage 1 (Feature Extraction):** Backbone frozen (`requires_grad=False`), training classifier head only with Adam (`lr=1e-3`, max 10 epochs, patience 3 on validation macro-F1).
  - **Stage 2 (Fine-Tuning):** Last 3 inverted residual blocks unfrozen (`features[-3:]`), BatchNorm layers of frozen blocks kept in `eval()` mode, discriminative learning rates (`1e-4` backbone, `1e-3` head) with Adam (max 20 epochs, patience 5 on validation macro-F1).
  - **Loss Function:** Class-weighted `CrossEntropyLoss` calculated strictly from the training set class distribution.

---

## 2. Augmentation Pipeline: What Was Applied & Why

### Transforms Applied
In the **Augmentation** configuration, the training dataset uses the following composition:

1. `transforms.RandomHorizontalFlip()`: Mirror reflection along the vertical axis (p=0.5).
2. `transforms.RandomVerticalFlip()`: Mirror reflection along the horizontal axis (p=0.5).
3. `transforms.RandomRotation(degrees=15)`: Slight in-plane rotation within $[-15^\circ, +15^\circ]$ to simulate arbitrary item orientation.
4. `transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`: Photometric perturbations to handle lighting variation and exposure differences.
5. `transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0))`: Random cropping covering 70% to 100% of the image area followed by bilinear interpolation to 224×224, simulating varying camera-to-waste distances and partial framing.
6. `transforms.ToTensor()`: Conversion from PIL image to PyTorch float tensor in range $[0.0, 1.0]$.
7. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`: Standard ImageNet channel standardization.

In the **No Augmentation** baseline configuration, the training transform consists solely of:
1. `transforms.ToTensor()`
2. `transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`

### Why Augment Only the Training Set?
Validation and test sets serve as proxies for real-world deployment data. Their purpose is to measure model performance on canonical, uncorrupted images. 

Augmenting the validation or test splits is bad practice because:
1. **Evaluation Integrity:** Perturbing test samples injects stochastic noise into the evaluation metric, creating moving-target benchmarks that cannot be compared across models.
2. **Distribution Shift:** Test images must reflect the true sensor distribution captured by a camera at inference time, not synthetic cropping or jitter artifacts.
3. **Training Regularization Purpose:** Augmentation functions as an implicit regularizer during optimization. It expands the effective training manifold by presenting distinct variations of each image at each epoch, preventing the network from memorizing background textures or specific pixel configurations.

---

## 3. Empirical Results

All numbers below are extracted directly from `reports/augmentation_summary.csv` and `reports/augmentation_runs.csv`.

### Summary Statistics (Mean ± Std over 3 Seeds)

| Metric | No Augmentation (N=3) | Augmentation (N=3) | Delta (Aug - NoAug) |
| :--- | :---: | :---: | :---: |
| **Macro-F1 Score** | **0.9001 ± 0.0064** | **0.9179 ± 0.0037** | **+0.0179** |
| **Overall Accuracy (%)** | **90.76% ± 0.46%** | **92.00% ± 0.31%** | **+1.23%** |
| **Cardboard F1** | 0.9504 ± 0.0053 | 0.9325 ± 0.0105 | -0.0179 |
| **Glass F1** | 0.8987 ± 0.0104 | 0.9238 ± 0.0038 | +0.0251 |
| **Metal F1** | 0.8976 ± 0.0136 | 0.9086 ± 0.0022 | +0.0109 |
| **Paper F1** | 0.9383 ± 0.0095 | 0.9371 ± 0.0000 | -0.0012 |
| **Plastic F1** | 0.8745 ± 0.0052 | 0.9003 ± 0.0256 | +0.0258 |
| **Trash F1** | 0.8407 ± 0.0287 | 0.9053 ± 0.0446 | +0.0646 |
| **Train/Val Loss Gap** | 0.2769 ± 0.0289 | 0.1918 ± 0.0273 | -0.0851 |
| **Training Time (s)** | 142.2s ± 24.9s | 290.9s ± 17.2s | +148.7s |

### Per-Run Detailed Results (`reports/augmentation_runs.csv`)

| Config | Seed | Val Acc | Val Macro-F1 | Cardboard F1 | Glass F1 | Metal F1 | Paper F1 | Plastic F1 | Trash F1 | Train Loss (Best) | Val Loss (Best) | Loss Gap | Train Time | Best Stage / Epoch |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Aug** | 42 | 0.9050 | 0.8988 | 0.9565 | 0.9079 | 0.8819 | 0.9274 | 0.8690 | 0.8500 | 0.0443 | 0.3191 | 0.2748 | 119.1s | Stage 2 / Ep 7 |
| **No Aug** | 43 | 0.9129 | 0.9070 | 0.9474 | 0.9007 | 0.9062 | 0.9444 | 0.8794 | 0.8636 | 0.0549 | 0.3040 | 0.2491 | 139.0s | Stage 2 / Ep 6 |
| **No Aug** | 44 | 0.9050 | 0.8944 | 0.9474 | 0.8874 | 0.9048 | 0.9432 | 0.8750 | 0.8085 | 0.0179 | 0.3246 | 0.3067 | 168.6s | Stage 2 / Ep 13 |
| **Aug** | 42 | 0.9235 | 0.9165 | 0.9391 | 0.9211 | 0.9104 | 0.9371 | 0.9275 | 0.8636 | 0.1271 | 0.2892 | 0.1621 | 271.0s | Stage 2 / Ep 12 |
| **Aug** | 43 | 0.9182 | 0.9152 | 0.9204 | 0.9281 | 0.9091 | 0.9371 | 0.8966 | 0.9000 | 0.0925 | 0.3082 | 0.2157 | 301.2s | Stage 2 / Ep 16 |
| **Aug** | 44 | 0.9182 | 0.9221 | 0.9381 | 0.9221 | 0.9062 | 0.9371 | 0.8767 | 0.9524 | 0.0788 | 0.2764 | 0.1976 | 300.5s | Stage 2 / Ep 20 |

---

## 4. Key Findings & Empirical Analysis

1. **Clear Macro-F1 Improvement:**
   - Mean validation macro-F1 increased from **0.9001 ± 0.0064** to **0.9179 ± 0.0037**, an improvement of **+0.0179** (+1.79 percentage points).
   - The variance across seeds was reduced by nearly half (std dropped from 0.0064 to 0.0037), indicating more stable optimization.

2. **Strict Separation Across Seeds:**
   - **Every augmented seed outperformed every non-augmented seed.**
   - The lowest performing augmented run (Seed 43: **0.9152**) was strictly higher than the best non-augmented run (Seed 43: **0.9070**).

3. **Substantial Reduction in Generalization Gap:**
   - The train/validation loss gap at the best checkpoint epoch dropped from **0.2769 ± 0.0289** down to **0.1918 ± 0.0273** (a reduction of **-0.0851**).
   - In the un-augmented runs, the model overfit rapidly to training samples, driving training loss down to 0.0179–0.0549 while validation loss plateaued around 0.3040–0.3246.
   - Augmentation maintained a healthy training loss (0.0788–0.1271) and lowered validation loss (0.2764–0.3082), confirming that the network learned more generalizable features.

4. **Training Time Trade-Off:**
   - Wall-clock training time approximately doubled: **142.2s ± 24.9s** for No Augmentation versus **290.9s ± 17.2s** for Augmentation (+148.7s).
   - This overhead stems from two sources: on-the-fly image transformations computed on CPU per batch, and longer training trajectories before early stopping triggered (best epochs were 12, 16, and 20 for Aug versus 7, 6, and 13 for No Aug).

5. **Per-Class Differential Effects:**
   - **Trash F1:** Showed the largest gain, rising from **0.8407 ± 0.0287** to **0.9053 ± 0.0446** (**+0.0646**). The minority class benefited most from synthetic view variations.
   - **Plastic F1:** Improved from **0.8745 ± 0.0052** to **0.9003 ± 0.0256** (**+0.0258**).
   - **Glass F1:** Improved from **0.8987 ± 0.0104** to **0.9238 ± 0.0038** (**+0.0251**).
   - **Metal F1:** Improved slightly from **0.8976 ± 0.0136** to **0.9086 ± 0.0022** (**+0.0109**).
   - **Paper F1:** Remained virtually identical at **0.9383 ± 0.0095** vs **0.9371 ± 0.0000** (**-0.0012**).
   - **Cardboard F1:** Slightly declined from **0.9504 ± 0.0053** to **0.9325 ± 0.0105** (**-0.0179**). Random crops and color jitter occasionally made flat brown cardboard textures resemble crumpled paper or dark trash items.

---

## 5. Model Selection Decision

- **Chosen Configuration:** **Augmentation**.
- **Final Deployment Model:** `models/mobilenetv2_augmentation_seed44.pt` (copied to `models/mobilenetv2_final.pt`).
- **Rationale:** 
  - Seed 44 of the augmented configuration achieved the highest validation performance:
    - **Validation Macro-F1:** **0.9221** (highest across all 6 runs)
    - **Validation Accuracy:** **91.82%**
    - **Validation Loss:** **0.2764** (lowest validation loss of all runs)
    - **Loss Gap:** **0.1976**
    - **Trash F1:** **0.9524**
  - Hash verification confirms `models/mobilenetv2_final.pt` is bit-for-bit identical to `models/mobilenetv2_augmentation_seed44.pt` (SHA256: `a00cc6b6ad1577c2a4a418a5044b1273744a24de70ad9441be8f185f12431673`).

---

## 6. Honest Caveats & Scientific Limitations

To maintain scientific integrity and avoid overstated claims:

1. **Small Sample of Seeds (N=3):**
   - Evaluating across 3 seeds provides an initial measure of variance, but 3 seeds offer limited statistical power (2 degrees of freedom). While the separation is clean across these seeds, higher seed counts (e.g., N=10) would be required for formal hypothesis testing.

2. **Single Validation Set Split:**
   - All comparisons were conducted on a single fixed 15% validation split (379 images). They do not represent repeated k-fold cross-validation across multiple folds.

3. **Minority Class Sample Size (Trash N=21):**
   - The validation set contains only **21 images of trash**. Because the denominator is small, a single misclassified sample shifts class recall and F1 by nearly 4.8 percentage points. This small sample size explains why trash exhibited the largest standard deviation (0.0446).

4. **Optimistic Bias of Validation-Based Model Selection:**
   - Selecting the single best seed (Seed 44) based on its validation score introduces a mild selection bias ("winner's curse"). The validation macro-F1 of 0.9221 may be slightly optimistic.
   - **The TEST set evaluation in Phase 6 provides the sole unbiased, definitive performance benchmark for the final model.**
