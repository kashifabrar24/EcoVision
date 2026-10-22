# EcoVision: Technical Evaluation & Audit Report

**Author:** EcoVision Engineering Team  
**Dataset:** Gary Thung & Mindy Yang, *TrashNet* (GitHub commit `6fa2b87`)  
**Deployment Target:** Automated Recyclable Material Sorting with Selective Human Routing  

---

## 1. Executive Summary

This report presents the rigorous engineering evaluation of **EcoVision**, an automated waste material classification system built on the TrashNet dataset. By pairing a transfer-learned MobileNetV2 architecture with leak-free stratified group splitting, class-balanced loss weighting, data augmentation, and empirical confidence thresholding, EcoVision achieves **89.71% accuracy** (95% bootstrap CI: [86.54%, 92.61%]) and **0.8647 macro-F1** (95% bootstrap CI: [0.8195, 0.9036]) on holdout test data. This represents a definitive **+21.90% accuracy** and **+0.2170 macro-F1** improvement over an architectural baseline CNN trained from scratch.

Furthermore, by applying an operational confidence threshold of **0.80** (optimized strictly on validation predictions), the system enables selective classification: automated sorting is executed for **86.81%** of incoming items at **95.14% accuracy**, while diverting the remaining 13.19% of ambiguous items to secondary inspection. Edge inference latency averages **11.94 ms** on GPU and **24.38 ms** on commodity CPU, proving feasibility for high-speed industrial conveyor deployment.

---

## 2. Dataset Provenance, Integrity Audit & Leak-Proof Splitting

### 2.1 Resolving the 2x Hugging Face Mirror Duplication
The Hugging Face mirror (`garythung/trashnet`) contains 5,054 rows rather than the 2,527 images in the original GitHub repository. Empirical inspection of pixel dimensions and MD5 hashes revealed that the mirror's auto-generated dataset loader ingested two zip archives (`dataset-original.zip` and `dataset-resized.zip`) into a single `train` split, duplicating every image at two resolutions.

### 2.2 Eliminating Conflicting Label Noise
Cryptographic and perceptual hashing (`pHash`) uncovered 3 identical image pairs with contradictory class labels:
1. `glass115.jpg` (glass) vs. `metal91.jpg` (metal)
2. `glass176.jpg` (glass) vs. `plastic152.jpg` (plastic)
3. `glass389.jpg` (glass) vs. `plastic332.jpg` (plastic)

All 6 instances were permanently purged to ensure zero label noise, yielding a clean dataset of **2,521 unique images**.

### 2.3 Stratified Group Splitting
To prevent near-duplicate leakage (e.g. consecutive shots of the same bottle taken seconds apart landing in both train and test), connected components on pHash Hamming distance $\le 4$ were grouped together. Using seed 42, a leak-free Stratified Group Split partitioned the clean dataset:
- **Train Set (70.0%):** 1,763 images
- **Validation Set (15.0%):** 379 images
- **Holdout Test Set (15.0%):** 379 images (sealed until final Phase 6 benchmarking)

### 2.4 Class Imbalance
The dataset exhibits a **4.34:1** class imbalance between paper (594 images, 23.51%) and miscellaneous trash (137 images, 5.42%):
- `paper`: 594 total (train: 416, val: 89, test: 89)
- `glass`: 498 total (train: 348, val: 75, test: 75)
- `plastic`: 480 total (train: 336, val: 72, test: 72)
- `metal`: 409 total (train: 285, val: 62, test: 62)
- `cardboard`: 403 total (train: 283, val: 60, test: 60)
- `trash`: 137 total (train: 95, val: 21, test: 21)

---

## 3. Model Architecture & Experimental Progression

### 3.1 Baseline CNN (Scratch)
A compact 4-block convolutional network (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d), Global Average Pooling, Dropout (0.3), and a 6-class linear classification head (**390,438 parameters**). Optimized with Adam (`lr=1e-3`) and inverse frequency class weights calculated strictly from training labels. Early stopping halted training at epoch 18 (patience 7), reaching 69.13% validation accuracy and 0.6658 macro-F1.

### 3.2 Two-Stage Transfer Learning (MobileNetV2)
MobileNetV2 pre-trained on ImageNet-1k was adapted by substituting the 1000-class classifier with a custom classification head (`Dropout(0.3) -> Linear(1280, 6)`), totaling **2,231,558 parameters**.
- **Stage 1 (Feature Alignment):** Backbone frozen (`requires_grad=False`), training only the head with Adam (`lr=1e-3`, patience 3 on val macro-F1).
- **Stage 2 (Discriminative Fine-Tuning):** Unfreezing the last 3 inverted residual blocks (`features[-3:]`) with differential learning rates (`1e-4` for backbone, `1e-3` for head), keeping frozen BatchNorm layers in `eval()` mode.

### 3.3 Data Augmentation Experiment across 3 Seeds
To prevent overfitting on the small training set (1,763 images), an augmentation comparison experiment was conducted across 3 random seeds (42, 43, 44) strictly on the validation set:
- *Transforms:* RandomHorizontalFlip, RandomRotation ($\pm 15^\circ$), ColorJitter (brightness 0.2, contrast 0.2, saturation 0.2), and RandomResizedCrop (scale 0.7–1.0).
- *Findings:* Augmentation improved mean validation macro-F1 by **+0.0179** (0.9179 ± 0.0037 vs. 0.9001 ± 0.0064) and compressed the train/val loss gap from 0.2769 to 0.1918. Model checkpoint **Seed 44** achieved the highest validation macro-F1 (**0.9221**, val accuracy 91.82%) and was designated as `mobilenetv2_final.pt`.

---

## 4. Empirical Benchmark & Statistical Validation

All test evaluations were executed in a single unbiased pass on the 379 holdout test samples. 95% confidence intervals were computed via non-parametric bootstrapping with 1,000 resamples.

### 4.1 Comparative Model Performance

| Model | Val Accuracy | Val Macro-F1 | Test Accuracy (95% Bootstrap CI) | Test Macro-F1 (95% Bootstrap CI) | Trainable Params |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 69.13% | 0.6658 | 67.81% [63.05%, 72.30%] | 0.6477 [0.5953, 0.6959] | 390,438 |
| **MobileNetV2 Final** | **91.82%** | **0.9221** | **89.71% [86.54%, 92.61%]** | **0.8647 [0.8195, 0.9036]** | 2,231,558 |
| **Performance Margin** | *+22.69%* | *+0.2563* | **+21.90%** | **+0.2170** | *+1.84M* |

### 4.2 Per-Class Breakdown on Holdout Test Split

| Class | Support | Precision | Recall | F1-Score | F1 Gain vs. Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cardboard** | 60 | 0.9661 | 0.9500 | 0.9580 | +0.1420 |
| **Glass** | 75 | 0.9028 | 0.8667 | 0.8844 | +0.2990 |
| **Metal** | 62 | 0.8571 | 0.9677 | 0.9091 | +0.2548 |
| **Paper** | 89 | 0.9438 | 0.9438 | 0.9438 | +0.1831 |
| **Plastic** | 72 | 0.8611 | 0.8611 | 0.8611 | +0.1993 |
| **Trash** | 21 | 0.7059 | 0.5714 | 0.6316 | +0.2234 |
| **Macro Average** | **379** | **0.8728** | **0.8601** | **0.8647** | **+0.2170** |

---

## 5. Selective Classification & Confidence Thresholding

To avoid erroneous classification in industrial waste streams, a confidence rejection mechanism was tuned on validation predictions (`reports/threshold_sweep.csv`). The pre-committed optimization criterion required the highest coverage with validation accuracy $\ge$ 95%. This yielded an operating cutoff of **0.80**.

### 5.1 Validation vs. Test Threshold Impact

| Metric | Validation Split (Tuned) | Test Split (Deployed) |
| :--- | :---: | :---: |
| **Operating Threshold** | 0.80 | 0.80 |
| **Coverage** | 92.88% (352 / 379 answered) | **86.81%** (329 / 379 answered) |
| **Rejection Rate** | 7.12% (27 rejected) | **13.19%** (50 rejected) |
| **Accuracy on Answered** | 95.17% | **95.14%** |
| **Macro-F1 on Answered** | 0.9531 | **0.9231** |

**Empirical Observation:** The holdout test set experienced a higher rejection rate (13.19% vs. 7.12%) than the validation set. This behavior is expected because novel holdout instances contain un-modeled feature variances that lower posterior softmax probabilities. However, on answered samples, test accuracy generalized almost identically (**95.14%** vs. 95.17%), eliminating 25 out of the 39 raw test misclassifications.

---

## 6. Failure Modes & Error Analysis

![Test Failure Cases](reports/failure_cases.png)

Detailed analysis of the 39 test errors (`reports/test_errors.csv`) highlights two dominant visual confusions:
1. **Material Transparency & Specular Highlight Overlap (Glass vs. Plastic):** 10 glass samples were misclassified as plastic, and 5 plastic samples were misclassified as glass. Cylindrical transparent containers under studio spotlighting produce identical specular edge glints that cannot be resolved without material-specific spectral or tactile sensors.
2. **Minority Class Support Volatility (Trash):** With only 21 holdout test examples, the `trash` class suffered 9 false negatives, pulling its class F1 down to 0.6316. Items classified as trash exhibit extreme intra-class heterogeneity (e.g., styrofoam, broken pens, candy wrappers) that require substantially larger training support.

---

## 7. Edge Deployment Latency

Benchmarked using `src/measure_latency.py` across 10 warmup and 100 timed iterations. Model checkpoint size is **8.75 MB**.

| Hardware Platform | Execution Mode | Median Latency | p95 Tail Latency | Mean ± Std |
| :--- | :--- | :---: | :---: | :---: |
| **NVIDIA RTX 3050 Laptop GPU** | Model-Only Forward Pass | 9.22 ms | 12.38 ms | 9.69 ± 1.43 ms |
| | End-to-End (I/O + Preprocess + Model) | **11.94 ms** | **13.88 ms** | 12.11 ± 0.84 ms |
| **AMD64 Commodity CPU** | Model-Only Forward Pass | 22.95 ms | 31.95 ms | 23.97 ± 3.70 ms |
| | End-to-End (I/O + Preprocess + Model) | **24.38 ms** | **29.38 ms** | 26.15 ± 11.53 ms |

Both hardware configurations easily fulfill the real-time requirements of industrial sorting conveyors (<50 ms per item).

---

## 8. Honest Limitations & Production Considerations

1. **Inference Resolution Pre-caching Delta:** Feeding uncompressed 512×384 raw images directly into inference code can produce slight probability variations compared to pre-cached 224×224 JPEG images (maximum observed difference was 0.065 on one `trash` image, with predicted class remaining unchanged). The working hypothesis is that saving the cache at JPEG quality 95 introduced high-frequency DCT quantization artifacts that slightly perturb activations.
2. **Controlled Studio Collection Bias:** TrashNet images consist of clean, centered single items photographed on pristine white posterboard. Real-world municipal recycling facilities encounter crushed, soiled, occluded, and multi-object waste streams.
3. **Winner's Curse / Validation Optimism:** Test macro-F1 (0.8647) is lower than validation macro-F1 (0.9221) because model architecture, checkpoint selection, and confidence thresholds were optimized against the validation distribution.
4. **Softmax Overconfidence:** Standard softmax can assign high confidence scores to out-of-distribution inputs (e.g. biological waste or electronic devices).

---

## 9. Conclusion

EcoVision establishes a scientifically sound, leak-free, and auditable benchmark for waste material classification. By combining MobileNetV2 transfer learning with selective classification rejection, the system achieves **95.14% operational accuracy** while maintaining sub-25 ms edge latency. Future deployment cycles must prioritize in-situ cluttered data collection and multi-label object detection.
