# EcoVision Requirements & Deliverables Verification Checklist

This document maps every core requirement, engineering deliverable, and brownie point specification to the exact source file, test artifact, or report that verifies its implementation.

---

## 1. Core Engineering & Architecture Requirements

| Requirement Description | Status | Evidence / Proving File | Technical Verification Notes |
| :--- | :---: | :--- | :--- |
| **Deterministic Seed (Seed 42)** | **Done** | `src/dedupe_split.py`, `src/train_baseline.py`, `src/train_transfer.py`, `src/train_augmentation.py` | Python random, NumPy, and PyTorch seeds set to 42 across all data partitioning, baseline, and transfer pipelines. |
| **Windows Safety** | **Done** | All `src/*.py`, `app.py`, `space/app.py` | Strictly uses `pathlib.Path` for cross-platform filesystem operations and enforces `num_workers=0` on PyTorch DataLoaders to prevent Windows multiprocessing spawn errors. |
| **Zero Invented Numbers Rule** | **Done** | `reports/*.csv`, `reports/*.json`, `models/*.json`, `README.md`, `REPORT.md` | All reported performance metrics, class counts, confidence intervals, latencies, and parameter counts originate from automated script outputs in `reports/`. |
| **Console Output Hygiene** | **Done** | All `src/*.py` | Console outputs formatted concisely without terminal spam or unbounded iteration dumps. |
| **Documentation Integrity (EXPLAINED.md)** | **Done** | `docs/EXPLAINED.md` | Every source file, configuration manifest, and report document possesses a dedicated section detailing what it does, why it exists, and three technical interview Q&As. |

---

## 2. Phase 1 & 2: Data Audit, Deduplication & Leak-Free Splitting

| Requirement Description | Status | Evidence / Proving File | Technical Verification Notes |
| :--- | :---: | :--- | :--- |
| **TrashNet Data Provenance** | **Done** | `src/download_dataset.py`, `docs/DATASET.md` | Dataset tracked from Gary Thung and Mindy Yang's repository (commit `6fa2b87`, 2,527 images) and Hugging Face mirror (`garythung/trashnet`, 5,054 rows). |
| **Resolve 2x Mirror Duplication** | **Done** | `src/data_audit.py`, `reports/discrepancy_summary.json` | Proven empirically in code: Hugging Face auto-loader combined `dataset-original.zip` and `dataset-resized.zip` into one split. |
| **Contradictory Label Purge** | **Done** | `src/dedupe_split.py`, `reports/removed_images.csv` | Cryptographic MD5 matching identified 3 pairs (6 images) with conflicting labels (`glass115`/`metal91`, `glass176`/`plastic152`, `glass389`/`plastic332`), permanently deleted to leave 2,521 clean images. |
| **Near-Duplicate Grouping** | **Done** | `src/dedupe_split.py`, `reports/near_duplicates_metrics.json` | Perceptual hashing (`pHash`) with connected components at Hamming distance $\le 4$ clusters near-identical object poses into atomic groups. |
| **Stratified Group Split (70/15/15)** | **Done** | `src/dedupe_split.py`, `reports/split_summary.csv` | Partitioned into 1,763 train (70%), 379 validation (15%), and 379 test (15%) images with balanced class distributions and zero group contamination across splits. |
| **Standardized 224x224 Pre-caching** | **Done** | `src/dedupe_split.py`, `data/processed/` | Images resized directly to 224×224 (Bilinear interpolation, NO center crop) and pre-cached into `data/processed/` subdirectories to prevent training I/O bottlenecks. |

---

## 3. Phase 3 & 4: Baseline CNN & Two-Stage Transfer Learning

| Requirement Description | Status | Evidence / Proving File | Technical Verification Notes |
| :--- | :---: | :--- | :--- |
| **Baseline CNN from Scratch** | **Done** | `src/models.py`, `src/train_baseline.py`, `reports/baseline_vs_transfer_val.csv` | 4-block CNN (390,438 parameters) with GAP and Dropout(0.3). Achieved 69.13% val accuracy and 0.6658 macro-F1 with early stopping (patience 7). |
| **Inverse Frequency Class Weighting** | **Done** | `src/train_baseline.py`, `src/train_transfer.py` | Balanced weights computed strictly from the 1,763 training samples ($w_c = \frac{N}{C \cdot N_c}$) to penalize errors on minority trash (95 train images) vs. majority paper (416 train images). |
| **Pristine Holdout Test Set** | **Done** | `src/train_baseline.py`, `src/train_transfer.py`, `src/train_augmentation.py` | Test set sealed during all modeling, hyperparameter tuning, and early stopping decisions. |
| **MobileNetV2 Transfer Learning** | **Done** | `src/models.py`, `src/train_transfer.py`, `models/mobilenetv2_transfer.pt` | Adapted ImageNet-pretrained MobileNetV2 with custom head (`Dropout(0.3) -> Linear(1280, 6)`, 2,231,558 total parameters). |
| **Two-Stage Fine-Tuning Discipline** | **Done** | `src/train_transfer.py`, `reports/transfer_history.csv` | Stage 1 trains only classification head (`lr=1e-3`, backbone frozen); Stage 2 unfreezes last 3 inverted residual blocks with discriminative learning rate (`1e-4` backbone, `1e-3` head) and frozen BatchNorm layers kept in `eval()` mode. |

---

## 4. Phase 5 & 6: Data Augmentation & Final Holdout Evaluation

| Requirement Description | Status | Evidence / Proving File | Technical Verification Notes |
| :--- | :---: | :--- | :--- |
| **Augmentation Experiment (3 Seeds)** | **Done** | `src/train_augmentation.py`, `reports/augmentation_summary.csv`, `reports/augmentation_runs.csv` | Compared un-augmented vs. augmented configurations across seeds 42, 43, and 44 strictly on validation. Augmentation improved mean macro-F1 by +0.0179 (0.9179 vs. 0.9001) and narrowed loss gap from 0.2769 to 0.1918. |
| **Final Model Selection** | **Done** | `src/train_augmentation.py`, `models/mobilenetv2_final.pt` | Augmentation Seed 44 achieved top validation macro-F1 (0.9221, val acc 91.82%) and was saved as final production weights (8.75 MB). |
| **Single-Pass Holdout Test Audit** | **Done** | `src/evaluate_test.py`, `reports/test_comparison.csv` | Test set evaluated once in Phase 6: MobileNetV2 achieved 89.71% accuracy (+21.90% over baseline) and 0.8647 macro-F1 (+0.2170 over baseline). |
| **Bootstrap 95% Confidence Intervals** | **Done** | `src/evaluate_test.py`, `reports/test_bootstrap_ci.json` | 1,000 resamples with replacement established empirical 95% CIs: Accuracy [86.54%, 92.61%], Macro-F1 [0.8195, 0.9036]. |
| **Error Diagnostics & Visualizations** | **Done** | `reports/test_errors.csv`, `reports/failure_cases.png`, `reports/test_confusion_matrix.png` | Comprehensive logging of all 39 test errors, identifying glass/plastic transparency overlap and minority trash false negatives. |

---

## 5. Phase 7: Confidence Threshold Optimization

| Requirement Description | Status | Evidence / Proving File | Technical Verification Notes |
| :--- | :---: | :--- | :--- |
| **Threshold Tuning on Validation Only** | **Done** | `src/tune_threshold.py`, `reports/threshold_sweep.csv`, `models/threshold.json` | Swept cutoffs 0.30–0.95 strictly on `reports/val_predictions.csv`. Pre-committed rule (highest coverage with val acc $\ge 95\%$) selected cutoff **0.80** (92.88% coverage, 95.17% accuracy). |
| **Test Set Verification with Threshold** | **Done** | `src/tune_threshold.py`, `reports/test_with_threshold.json` | Cutoff 0.80 applied once to test set: 86.81% coverage (329 answered, 50 rejected / 13.19%), accuracy on answered reached **95.14%** (macro-F1 0.9231), eliminating 25 out of 39 test errors. |
| **Honest Reporting of Rejection Disparity** | **Done** | `README.md`, `REPORT.md`, `docs/TEST_RESULTS.md` | Plainly reported that test set rejected more images (13.19%) than validation (7.12%) due to novel test distribution variance. |

---

## 6. Phase 8 & 9: Production Inference & Deployment Interfaces

| Requirement Description | Status | Evidence / Proving File | Technical Verification Notes |
| :--- | :---: | :--- | :--- |
| **Production CLI & Programmatic API** | **Done** | `src/predict.py` | CLI `python src/predict.py path/to/image.jpg` prints class, confidence, top-3, and "UNCERTAIN" if below threshold. `predict_image(pil_image)` returns full probability dict, predicted class, and uncertainty flag. |
| **Preprocessing Parity (NO Train-Serve Skew)** | **Done** | `src/predict.py` | Exact mirror of dataset caching: RGB conversion, direct resize to 224×224 (Bilinear, NO crop), ToTensor, ImageNet normalization. |
| **Class Export** | **Done** | `models/classes.json` | Saved in ImageFolder alphabetical order (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`). |
| **Hardware Latency Benchmarking** | **Done** | `src/measure_latency.py`, `reports/latency.json` | Profiled on GPU (RTX 3050: 9.22 ms model / 11.94 ms e2e) and CPU (22.95 ms model / 24.38 ms e2e) across 10 warmup and 100 timed runs with `torch.cuda.synchronize()`. |
| **Interactive Gradio Web Application** | **Done** | `app.py`, `demo_examples/` | Features `gr.Label` (all 6 class probabilities), decision status line, honest note on campus waste, and 6 registered test split examples (< 200 KB each). Tested on `http://127.0.0.1:7860` with `gradio_client`. |
| **Self-Contained Hugging Face Space** | **Done** | `space/app.py`, `space/requirements.txt`, `space/README.md`, `space/model.pt` | Fully isolated directory (8.93 MB < 100 MB limit) running entirely on CPU (`map_location="cpu"`), importing zero code from `src/`, with CPU PyTorch wheel index on first line and YAML metadata in `space/README.md`. |

---

## 7. Phase 10 & 11: Comprehensive Documentation & Audits

| Requirement Description | Status | Evidence / Proving File | Technical Verification Notes |
| :--- | :---: | :--- | :--- |
| **Architecture Diagram** | **Done** | `docs/architecture.png`, `docs/architecture.mmd` | Rendered with high DPI and zero overlapping text or intersecting lines; depicts the complete lifecycle from dual archive ingestion to edge serving. |
| **Root README** | **Done** | `README.md` | Executive summary, results table with 95% bootstrap CIs, dataset provenance, architecture diagram, methodology, reproduce-from-scratch instructions, latency table, threshold analysis, failure cases, technical caveats, and dataset citation. |
| **Technical Evaluation Report** | **Done** | `REPORT.md` | Formal 2-page evaluation document detailing data cleaning, experimental progression, statistical significance, selective classification, error analysis, edge latency, and operational limitations. |
| **30 Machine Learning Interview Q&As** | **Done** | `docs/INTERVIEW_QA.md` | 30 conversational, deep questions and answers referencing EcoVision's empirical numbers and architectural decisions. |
| **Pinned Dependencies** | **Done** | `requirements.txt` | Pinned to exact virtual environment versions; verified with `pip check` (0 broken requirements). |
| **Relative Image Links Enforcement** | **Done** | All `.md` files | Verified via automated audit: zero absolute `c:/Users` or `file:///` paths across the entire repository. |
| **Fresh-Clone Verification** | **Done** | Git clone test | Verified clean clone into temp directory with `python -m compileall src app.py` (0 syntax errors) and confirmed all referenced files exist. |
