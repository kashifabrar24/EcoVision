# Final Holdout Test Evaluation & Confidence Threshold Report (TEST_RESULTS.md)

## 1. Executive Summary & Test Split Disclosure

In Phase 6, the holdout test set (**379 images**) was evaluated for the **first and only time**. In strict compliance with scientific standards, no retraining, architecture changes, hyperparameter adjustments, or model re-selection were made based on test results.

- **Evaluated Models:**
  1. `BaselineCNN` (`models/baseline_cnn.pt`): 4-layer convnet trained from scratch (~390k parameters).
  2. `MobileNetV2 Final` (`models/mobilenetv2_final.pt`): ImageNet-pretrained MobileNetV2 fine-tuned with data augmentation (Seed 44 from Phase 5).
- **Comparison to Validation Performance:**
  - **The test performance is lower than the validation performance.**
  - Validation Macro-F1: **0.9221** | Test Macro-F1: **0.8647** ($\Delta = \mathbf{-0.0574}$)
  - Validation Accuracy: **91.82%** | Test Accuracy: **89.71%** ($\Delta = \mathbf{-2.11\%}$)
  - As anticipated in Phase 5 caveats, selecting the top-performing seed on validation introduces mild selection bias ("winner's curse"). Furthermore, the unweighted macro-F1 was heavily penalized by the extreme minority `trash` class (21 images), where F1 dropped from 0.9524 in validation to 0.6316 in test.
  - Nevertheless, MobileNetV2 Final decisively outperformed the scratch baseline by **+21.90% accuracy** and **+0.2170 macro-F1**.

---

## 2. Test Results Benchmark (with 95% Bootstrap CIs)

All metrics below are drawn directly from `reports/test_metrics.json`, `reports/test_comparison.csv`, and `reports/test_bootstrap_ci.json` (1,000 empirical bootstrap resamples, seed 42).

### Global Metrics Comparison

| Model | Test Accuracy | 95% Bootstrap CI (Accuracy) | Test Macro-F1 | 95% Bootstrap CI (Macro-F1) |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline CNN** | 67.81% | [63.05%, 72.30%] | 0.6477 | [0.5953, 0.6959] |
| **MobileNetV2 Final** | **89.71%** | **[86.54%, 92.61%]** | **0.8647** | **[0.8195, 0.9036]** |
| **Delta (Final - Baseline)** | **+21.90%** | — | **+0.2170** | — |

*Note: The 95% bootstrap confidence intervals for Baseline CNN and MobileNetV2 Final have zero overlap on both accuracy and macro-F1, confirming a statistically significant real-world advantage.*

### Per-Class Test Performance Comparison

| Class | Support | Baseline Precision | Baseline Recall | Baseline F1 | Final Precision | Final Recall | Final F1 | Delta F1 (Final - Baseline) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cardboard** | 60 | 0.7846 | 0.8500 | 0.8160 | **0.9661** | **0.9500** | **0.9580** | **+0.1420** |
| **Glass** | 75 | 0.7500 | 0.4800 | 0.5854 | **0.9028** | **0.8667** | **0.8844** | **+0.2990** |
| **Metal** | 62 | 0.5300 | 0.8548 | 0.6543 | **0.8571** | **0.9677** | **0.9091** | **+0.2548** |
| **Paper** | 89 | 0.8378 | 0.6966 | 0.7607 | **0.9438** | **0.9438** | **0.9438** | **+0.1831** |
| **Plastic** | 72 | 0.7031 | 0.6250 | 0.6618 | **0.8611** | **0.8611** | **0.8611** | **+0.1993** |
| **Trash** | 21 | 0.3571 | 0.4762 | 0.4082 | **0.7059** | **0.5714** | **0.6316** | **+0.2234** |

---

## 3. Phase 7: Confidence Thresholding (Selective Prediction)

To safeguard automated waste sorting systems against incorrect routing, Phase 7 implemented a selective prediction mechanism (rejecting low-confidence classifications as "uncertain").

### Selection Rule & Validation Sweep (`reports/threshold_sweep.csv`)
- **Fixed-in-Advance Selection Rule:** Select the threshold with the **highest coverage** such that **validation accuracy on answered images is $\ge$ 0.95**.
- **Chosen Threshold:** **0.80** (selected strictly from `reports/val_predictions.csv`).

| Sweep Threshold | Val Coverage (%) | Val Accuracy (%) | Val Macro-F1 | Answered / Total | Rejected |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.70 | 95.25% | 94.74% | 0.9490 | 361 / 379 | 18 |
| 0.75 | 94.46% | 94.69% | 0.9485 | 358 / 379 | 21 |
| **0.80 (Chosen)** | **92.88%** | **95.17%** | **0.9531** | **352 / 379** | **27** |
| 0.85 | 90.24% | 96.49% | 0.9646 | 342 / 379 | 37 |
| 0.90 | 86.81% | 96.96% | 0.9682 | 329 / 379 | 50 |
| 0.95 | 80.47% | 97.05% | 0.9681 | 305 / 379 | 74 |

### Application to Holdout Test Set (`reports/test_with_threshold.json`)
Applying threshold **0.80** once to the holdout test set without modification yielded:

- **Test Coverage:** **86.81%** (329 answered out of 379 images; 50 rejected as uncertain)
- **Test Accuracy on Answered:** **95.14%** (compared to raw test accuracy of 89.71%, a gain of **+5.43%**)
- **Test Macro-F1 on Answered:** **0.9231** (compared to raw test macro-F1 of 0.8647, a gain of **+0.0584**)
- **Rejected Fraction:** **13.19%** (50 images routed to manual inspection or secondary sorting)

*Key Insight:* The confidence threshold successfully removed 23 of the 39 test errors (while rejecting 27 correct images), pushing operational accuracy on answered images from 89.71% to 95.14% while retaining 86.81% automated throughput.

---

## 4. Per-Class Error Analysis

Out of 379 test samples, MobileNetV2 Final produced **39 misclassifications** (`reports/test_errors.csv`, visualized in `reports/failure_cases.png`):

1. **Glass vs. Plastic Confusion (10 glass errors, 4 plastic errors):**
   - *Observation:* 6 glass bottles were predicted as plastic (`glass13`, `glass210`, `glass221`, `glass254`, `glass293`, `glass305`), and 4 plastic containers were predicted as glass (`plastic196`, `plastic221`, `plastic364`, `plastic369`).
   - *Hypothesis:* Both materials share transparent, cylindrical geometries with specular reflections against white posterboard. Without tactile or weight feedback, optical 2D features alone struggle to discern clear PET plastic from clear soda-lime glass.

2. **Glass & Metal Confusion (4 glass errors, 2 metal errors):**
   - *Observation:* 4 glass items were predicted as metal (`glass170`, `glass309`, `glass385`, `glass410`), and 2 metal items were predicted as glass (`metal272`, `metal47`).
   - *Hypothesis:* Glass jars often possess metallic screw-on lids or dark tinting, and polished aluminum cans display sharp specular highlights that mimic glass glossiness.

3. **Paper vs. Plastic Confusion (3 paper errors, 2 plastic errors):**
   - *Observation:* 3 paper items were predicted as plastic (`paper171`, `paper314`, `paper53`), and 2 plastic items were predicted as paper (`plastic12`, `plastic465`).
   - *Hypothesis:* Glossy magazine cutouts and laminated paper food packaging have surface sheens similar to smooth plastic films, while white matte plastic resembles flat printer paper.

4. **The Minority `Trash` Challenge (9 trash errors, 5 false trash predictions):**
   - *Observation:* 9 of the 21 trash items were misclassified: 3 as metal (`trash128`, `trash35`, `trash50`), 3 as paper (`trash52`, `trash82`, `trash93`), 2 as cardboard (`trash135`, `trash14`), and 1 as glass (`trash32`). In reverse, 2 paper, 2 plastic, and 1 cardboard were predicted as trash.
   - *Hypothesis:* In TrashNet, the `trash` category is not a distinct physical material; it is a miscellaneous residual class comprising composite items (snack bags with foil linings, dirty cardboard cups, discarded batteries, plastic pens). When an item contains predominantly cardboard or foil parts, the convolutional filters detect those specific material textures.

---

## 5. Scientific Limitations & Engineering Caveats

To ensure honest deployment guidance and prevent misrepresentation of capability:

1. **Small Test Set ($N=379$):**
   - While sufficient for establishing clear superiority over the baseline, 379 images represent a modest statistical sample. Per-class sub-samples range from 21 to 89 images.
2. **Extreme Minority Sample Noise (Trash $N=21$):**
   - With only 21 test samples, a single error shifts trash recall and F1 by approximately 4.76 percentage points. The reported trash test F1 of 0.6316 carries wide variance and should be interpreted with caution.
3. **Studio Photography Bias (Domain Shift Risk):**
   - All TrashNet photos feature individual waste objects placed against uniform white posterboard under consistent indoor lighting. Real recycling conveyor belts or consumer disposal bins exhibit heavy occlusions, cluttered multi-object piles, crushed shapes, and dirty backgrounds. Performance in in-the-wild deployment is expected to degrade unless fine-tuned on real-world stream data.
4. **Near-Duplicate Grouping Limitations:**
   - Although Phase 2 implemented Stratified GROUP Splitting at perceptual hash distance $\le 4$ to minimize near-duplicate leakage, near-duplicate leakage is not fully excluded, and subtle background cues or uncaptured camera bursts from the original collection could still share slight distributional similarities across splits.
5. **Softmax Overconfidence:**
   - Standard softmax outputs are notoriously uncalibrated. Several misclassified samples exhibited high confidence scores ($>0.90$) due to extreme logit scaling. While confidence thresholding at 0.80 removed 23 of 39 errors (58.97%), temperature scaling or conformal prediction should be explored in future edge deployment iterations.
