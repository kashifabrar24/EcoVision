# EcoVision Requirements & Deliverables Checklist

This document maps each requirement of SC-AIML Task 1 (Waste Image Classification Pipeline) to the file that shows it, with an honest status. Every number comes from a file in `reports/` or `models/`.

Status key: **Done** = implemented and verified; **Done (with caveat)** = implemented, with a limitation stated in the notes.

---

## 1. Task Requirements (from the task description)

### Base Requirements

| Task requirement | Status | Evidence | Notes |
| :--- | :---: | :--- | :--- |
| Document dataset source, version and images per class | **Done** | `README.md` (Dataset & Provenance), `docs/DATASET.md`, `reports/split_summary.csv` | TrashNet by Gary Thung and Mindy Yang, GitHub commit `6fa2b87` (2,527 photos), plus the Hugging Face mirror (5,054 rows = original and resized archives of the same photos). After removing 6 conflicting-label images: 2,521 images. |
| Inspect the dataset and point out class imbalance | **Done** | `README.md`, `reports/split_summary.csv` | Imbalance ratio 4.34:1 (paper 594 vs trash 137). |
| Reproducible 70/15/15 splits, fixed seed, no duplicates across splits | **Done (with caveat)** | `src/dedupe_split.py`, `reports/split_summary.csv`, `reports/removed_images.csv` | Seed 42. Exact duplicates were removed (MD5) and near-duplicates were grouped by perceptual hash (Hamming distance <= 4) so each group stays in one split. Test filenames were checked against train and validation (0 overlapping files). Caveat: near-duplicate leakage cannot be fully excluded. |
| Train a baseline CNN or fine-tune a pretrained model | **Done** | `src/train_baseline.py`, `src/train_transfer.py`, `src/train_augmentation.py`, `models/` | Both were done: a baseline CNN from scratch and a two-stage fine-tuned MobileNetV2. |
| Evaluate with accuracy, per-class precision, recall, F1 and a confusion matrix | **Done** | `src/evaluate_test.py`, `reports/test_metrics.json`, `reports/test_comparison.csv`, `reports/test_confusion_matrix.png`, `README.md` (per-class table) | Test set (379 images) evaluated once. Accuracy 89.71%, macro-F1 0.8647. |
| Inference script or API returning class and confidence | **Done** | `src/predict.py`, `app.py`, `space/streamlit_app.py` | CLI, a Python function `predict_image`, a local Gradio app and a Streamlit app. |
| Include failure cases and explain why the model gets them wrong | **Done (with caveat)** | `reports/failure_cases.png`, `reports/test_errors.csv`, `README.md`, `docs/TEST_RESULTS.md` | 39 errors. Most frequent confusions: glass to plastic (6), glass to metal (4), plastic to glass (4), paper to plastic (3), trash to metal (3), trash to paper (3). The explanations are plausible hypotheses and were not tested. |

### Brownie Points

| Task brownie point | Status | Evidence | Notes |
| :--- | :---: | :--- | :--- |
| Compare a baseline CNN with transfer learning | **Done** | `reports/test_comparison.csv`, `README.md` | Test accuracy 67.81% (baseline) vs 89.71% (final MobileNetV2), with 95% bootstrap confidence intervals. |
| Data augmentation and its effect | **Done** | `src/train_augmentation.py`, `reports/augmentation_runs.csv`, `reports/augmentation_summary.csv` | Three seeds (42, 43, 44), validation only. Mean macro-F1 0.9179 with augmentation vs 0.9001 without. Three seeds is a small sample. |
| Confidence threshold tuning | **Done** | `src/tune_threshold.py`, `reports/threshold_sweep.csv`, `models/threshold.json`, `reports/test_with_threshold.json` | Chosen on validation only (0.80). On the test set, 86.81% of images were answered at 95.14% accuracy. |
| Inference latency measurement | **Done** | `src/measure_latency.py`, `reports/latency.json` | GPU about 12 ms and CPU about 24 ms end-to-end (median). |
| Error analysis by class and limitations for real campus waste | **Done (with caveat)** | `README.md`, `REPORT.md`, `docs/TEST_RESULTS.md` | Per-class results and limitations are documented. The model was not tested on real campus photos. |

### Deliverables

| Task deliverable | Status | Evidence |
| :--- | :---: | :--- |
| Dataset preparation code | **Done** | `src/download_dataset.py`, `src/data_audit.py`, `src/dedupe_split.py` |
| Training and inference code | **Done** | `src/train_*.py`, `src/models.py`, `src/predict.py` |
| Evaluation report | **Done** | `REPORT.md`, `docs/TEST_RESULTS.md` |
| Architecture diagram | **Done** | `docs/architecture.png`, `docs/architecture.mmd` |
| README | **Done** | `README.md` |
| Runnable demo | **Done** | Live Streamlit demo: https://ecovision-evzcs2cqdcwsrhk3fvdbxu.streamlit.app ; local demo: `python app.py` or `streamlit run space/streamlit_app.py` |

---

## 2. Engineering Practices

| Practice | Status | Evidence | Notes |
| :--- | :---: | :--- | :--- |
| Fixed seed (42) | **Done** | `src/dedupe_split.py`, `src/train_*.py` | Python, NumPy and PyTorch seeds set to 42. The augmentation experiment also uses seeds 43 and 44 on purpose. |
| Windows-safe code | **Done** | `src/*.py`, `app.py` | Uses `pathlib` and `num_workers=0` in DataLoaders. |
| Numbers come from scripts | **Done** | `reports/*.csv`, `reports/*.json`, `models/*.json` | Reported numbers were generated by the scripts. |
| Documentation of every file | **Done** | `docs/EXPLAINED.md` | Each file has what it does, why it exists, and three interview Q&As. |
| Test set kept untouched until the final evaluation | **Done** | `src/train_*.py`, `src/evaluate_test.py` | Training, early stopping, model selection and threshold tuning used validation data only. |

---

## 3. Phase 1 and 2: Data Audit, Deduplication and Leakage-Reduced Splitting

| Item | Status | Evidence | Notes |
| :--- | :---: | :--- | :--- |
| TrashNet provenance | **Done** | `src/download_dataset.py`, `docs/DATASET.md` | GitHub commit `6fa2b87` (2,527 images) and Hugging Face mirror `garythung/trashnet` (5,054 rows). |
| Explained the 2x mirror duplication | **Done** | `src/data_audit.py`, `reports/discrepancy_summary.json` | The mirror contains both the original and resized archives of the same photos. |
| Conflicting-label removal | **Done** | `src/dedupe_split.py`, `reports/removed_images.csv` | 3 pairs (6 images) with identical content but different labels were removed, leaving 2,521 images. |
| Near-duplicate grouping | **Done (with caveat)** | `src/dedupe_split.py`, `reports/near_duplicates_metrics.json` | Perceptual hashing with connected components at Hamming distance <= 4. Reduces but does not guarantee the absence of near-duplicates across splits. |
| Stratified group split 70/15/15 | **Done** | `src/dedupe_split.py`, `reports/split_summary.csv` | 1,763 train, 379 validation, 379 test; each group stays within one split. |
| 224x224 cache | **Done** | `src/dedupe_split.py`, `data/processed/` | Direct resize to 224x224 (bilinear, no center crop), saved as JPEG (quality 95). |

---

## 4. Phase 3 and 4: Baseline CNN and Transfer Learning

| Item | Status | Evidence | Notes |
| :--- | :---: | :--- | :--- |
| Baseline CNN | **Done** | `src/models.py`, `src/train_baseline.py`, `reports/baseline_vs_transfer_val.csv` | 4 convolutional blocks, 390,438 parameters. Validation accuracy 69.13%, macro-F1 0.6658. |
| Class weights | **Done** | `src/train_baseline.py`, `src/train_transfer.py` | Inverse-frequency weights computed from the training split only. |
| MobileNetV2 transfer learning | **Done** | `src/models.py`, `src/train_transfer.py`, `models/mobilenetv2_transfer.pt` | ImageNet-pretrained backbone with a custom head, 2,231,558 parameters. |
| Two-stage fine-tuning | **Done** | `src/train_transfer.py`, `reports/transfer_history.csv` | Stage 1 trains the head only; stage 2 unfreezes the last 3 inverted residual blocks with a lower backbone learning rate. |

---

## 5. Phase 5 and 6: Augmentation and Final Test Evaluation

| Item | Status | Evidence | Notes |
| :--- | :---: | :--- | :--- |
| Augmentation experiment | **Done** | `src/train_augmentation.py`, `reports/augmentation_summary.csv`, `reports/augmentation_runs.csv` | Mean validation macro-F1 improved by 0.0179 across 3 seeds; the train/val loss gap went from 0.2769 to 0.1918. |
| Final model selection | **Done** | `models/mobilenetv2_final.pt` | Seed 44 had the best validation macro-F1 (0.9221). Because it was chosen on validation, validation scores are slightly optimistic. |
| Single test evaluation | **Done** | `src/evaluate_test.py`, `reports/test_comparison.csv` | Test accuracy 89.71% and macro-F1 0.8647 (lower than validation, as expected after selection on validation). |
| Bootstrap 95% confidence intervals | **Done** | `reports/test_bootstrap_ci.json` | 1,000 resamples: accuracy [86.54%, 92.61%], macro-F1 [0.8195, 0.9036]. |
| Error diagnostics and visualizations | **Done (with caveat)** | `reports/test_errors.csv`, `reports/failure_cases.png`, `reports/test_confusion_matrix.png` | All 39 errors logged. Most frequent: glass to plastic (6), glass to metal (4), plastic to glass (4). The trash class has the lowest F1 (0.6316) with only 21 test images. Reasons given for errors are hypotheses. |

---

## 6. Phase 7: Confidence Threshold

| Item | Status | Evidence | Notes |
| :--- | :---: | :--- | :--- |
| Threshold chosen on validation only | **Done** | `src/tune_threshold.py`, `reports/threshold_sweep.csv`, `models/threshold.json` | Rule fixed in advance: highest coverage with validation accuracy >= 95%. Selected 0.80 (92.88% coverage, 95.17% accuracy). |
| Applied once to the test set | **Done** | `reports/test_with_threshold.json` | 329 answered (86.81%), 50 rejected (13.19%), accuracy on answered 95.14%. The threshold removed 23 of the 39 errors and also rejected 27 correct images. |
| Honest reporting of the validation-test gap | **Done** | `README.md`, `REPORT.md`, `docs/TEST_RESULTS.md` | The test set rejected more images than validation (13.19% vs 7.12%). A likely reason is that validation was used for selection (untested hypothesis). |

---

## 7. Phase 8 and 9: Inference and Demo

| Item | Status | Evidence | Notes |
| :--- | :---: | :--- | :--- |
| Inference CLI and Python function | **Done** | `src/predict.py` | Prints class, confidence, top-3 and UNCERTAIN below the threshold. `predict_image` returns probabilities, class and an uncertain flag. |
| Preprocessing matches training | **Done (with caveat)** | `src/predict.py` | RGB, direct resize to 224x224, ImageNet normalization. Predictions on original 512x384 images can differ slightly from the cached JPEG (largest observed difference 0.065 on one trash image, class unchanged; cause untested). |
| Class names export | **Done** | `models/classes.json` | Alphabetical order. |
| Latency benchmark | **Done** | `src/measure_latency.py`, `reports/latency.json` | GPU (RTX 3050): 9.22 ms model-only, 11.94 ms end-to-end. CPU: 22.95 ms and 24.38 ms. |
| Local Gradio app | **Done** | `app.py`, `demo_examples/` | Shows all 6 class probabilities, a decision line and 6 test-split examples. |
| Live demo | **Done** | `space/streamlit_app.py`, `space/requirements.txt` | https://ecovision-evzcs2cqdcwsrhk3fvdbxu.streamlit.app (Streamlit Community Cloud). A Hugging Face Space was not used because Gradio Spaces now require a paid plan. |

---

## 8. Phase 10 and 11: Documentation and Audits

| Item | Status | Evidence | Notes |
| :--- | :---: | :--- | :--- |
| Architecture diagram | **Done** | `docs/architecture.png`, `docs/architecture.mmd` | Shows the pipeline from dataset audit to inference and demo. |
| README | **Done** | `README.md` | Results, per-class table, confusion matrix, dataset provenance, threshold analysis, latency, limitations and citation. |
| Evaluation report | **Done** | `REPORT.md` | Data, splits, models, results, error analysis, threshold, latency and limitations. |
| Interview Q&As | **Done** | `docs/INTERVIEW_QA.md` | 30 questions using the project's real numbers. |
| Pinned dependencies | **Done** | `requirements.txt` | Pinned to installed versions; `pip check` reported no broken requirements. |
| Relative links only | **Done** | All `.md` files | Link audit found 0 broken image paths and 0 absolute paths. |
| Fresh-clone check | **Done** | Git clone into a temp folder | `python -m compileall src app.py` passed and referenced files exist. |

---

## 9. Known Limitations

- The model was trained on clean, single-item photos on a plain background and was **not tested on real campus waste photos**.
- About 2.5k images in total; the trash class has only 137 images (21 in the test set), so its metrics are uncertain.
- Near-duplicate leakage is reduced by perceptual-hash grouping but cannot be fully excluded.
- Softmax confidence can be over-confident, so the threshold is not a guarantee of correctness.
- Test macro-F1 (0.8647) is lower than validation macro-F1 (0.9221) because the checkpoint was selected on validation; the test numbers are the honest estimate.
- Explanations of failure cases are hypotheses and were not tested.