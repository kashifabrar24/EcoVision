# EcoVision Codebase Guide (EXPLAINED.md)

This document provides a concise explanation of every file in the project, why it exists, and three interview questions with straightforward answers designed for technical review.

---

## 1. `.gitignore`

### What it does
Specifies intentionally untracked files that Git should ignore, such as virtual environment folders (`venv/`), raw and cached dataset directories (`data/`), python bytecode caches (`__pycache__/`), and environment variable files (`.env`).

### Why it exists
Prevents accidental commits of large datasets, system-specific virtual environment binaries, and sensitive secrets to version control, keeping the repository lightweight, clean, and safe.

### Interview Questions & Answers
1. **Q: Why shouldn't raw dataset files be committed to Git?**
   - **A:** Git is designed for tracking incremental source code changes, not large binary blobs. Storing images bloats the `.git` directory history irreversibly, making cloning slow and exceeding hosting limits.
2. **Q: Why ignore `__pycache__` and `.pyc` files?**
   - **A:** They are compiled CPython bytecode artifacts specific to the Python interpreter and machine architecture running them. They are regenerated automatically at runtime.
3. **Q: Why are model weight checkpoints excluded unless under 50MB?**
   - **A:** Large checkpoints (>50MB or >100MB) can violate GitHub file limits and unnecessarily increase repository size. Only small, final deployment weights should be tracked or hosted on Model Hubs.

---

## 2. `requirements.txt`

### What it does
Declares the exact external Python packages and dependencies required to reproduce the environment, including PyTorch with CUDA, Hugging Face `datasets`, `imagehash`, `scikit-learn`, `pandas`, `matplotlib`, and `gradio`.

### Why it exists
Ensures complete reproducibility across machines. Any reviewer or recruiter can recreate the exact development environment by executing `pip install -r requirements.txt`.

### Interview Questions & Answers
1. **Q: Why is `--extra-index-url https://download.pytorch.org/whl/cu124` specified?**
   - **A:** Standard PyPI wheels for PyTorch on Windows can default to CPU-only. The extra index URL points directly to NVIDIA CUDA 12.4 pre-built wheels to guarantee GPU acceleration.
2. **Q: What is the purpose of pinning minimum package versions (e.g. `>=2.2.0`)?**
   - **A:** It balances stability and compatibility, ensuring features like modern torchvision transforms and Hugging Face datasets APIs are present without breaking on patch updates.
3. **Q: How does a virtual environment interact with `requirements.txt`?**
   - **A:** A virtual environment isolates project packages from global system Python, preventing dependency conflicts when installing the exact packages listed in `requirements.txt`.

---

## 3. `src/__init__.py`

### What it does
Marks the `src/` directory as a regular Python package, enabling modular imports (e.g., `from src.check_gpu import main`) from root scripts and notebooks.

### Why it exists
Improves code organization and maintainability by structuring pipeline components into an importable namespace.

### Interview Questions & Answers
1. **Q: Is `__init__.py` strictly required in Python 3.3+?**
   - **A:** Python 3.3+ introduced implicit namespace packages without `__init__.py`, but explicit `__init__.py` files make packages explicit, avoid namespace collisions, and allow package-level initialization.
2. **Q: Can initialization code be put inside `__init__.py`?**
   - **A:** Yes, code inside `__init__.py` runs when the package is imported, but keeping it empty or minimal is best practice to avoid unexpected side effects.
3. **Q: How does package structure help project explainability?**
   - **A:** It cleanly separates domain code (preprocessing, modeling, evaluation) from root entry-point scripts (`app.py`, notebooks), making the project easy to navigate during code reviews.

---

## 4. `src/check_gpu.py`

### What it does
Inspects the system's hardware environment, validates whether PyTorch has CUDA acceleration available (`torch.cuda.is_available()`), queries device properties (model name, VRAM), and exits with an error if only CPU is detected.

### Why it exists
Enforces Rule 7 ("First check torch.cuda.is_available(). If False, tell me and stop. Do not silently train on CPU") so training is never started on CPU by mistake, which would be orders of magnitude slower.

### Interview Questions & Answers
1. **Q: Why is `torch.cuda.is_available()` returning False even if `nvidia-smi` shows a GPU?**
   - **A:** `nvidia-smi` only checks the OS driver level. If PyTorch was installed from CPU-only wheels or if the PyTorch CUDA runtime version does not match the driver capability, `torch.cuda.is_available()` will be False.
2. **Q: Why is monitoring dedicated VRAM important for the RTX 3050 Laptop GPU?**
   - **A:** The RTX 3050 has 6GB (or 4GB on older variants) of VRAM. Large batch sizes or high-resolution images can trigger `CUDA out of memory` (OOM) errors, so batch size (32) and image size (224) must fit comfortably within the memory budget.
3. **Q: Why exit with a non-zero exit code on failure in CLI scripts?**
   - **A:** Non-zero exit codes signal failure to automated pipelines, CI/CD runners, and shell scripts, halting subsequent pipeline steps immediately.

---

## 5. `src/test_env.py`

### What it does
Imports each required project package (`torch`, `torchvision`, `datasets`, `huggingface_hub`, `PIL`, `imagehash`, `sklearn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `gradio`, `tqdm`) and verifies its version and runtime availability.

### Why it exists
Serves as an automated sanity check for the virtual environment setup to guarantee all libraries are correctly installed before training or data processing begins.

### Interview Questions & Answers
1. **Q: Why test package imports programmatically instead of just running the main script?**
   - **A:** Programmatic verification isolates dependency and binary compatibility problems immediately with descriptive error logs, avoiding mid-pipeline failures after hours of downloading or preprocessing.
2. **Q: What is the benefit of dynamic importing via `importlib.import_module`?**
   - **A:** It allows looping over a list of package strings cleanly, catching import errors per package rather than failing fast on the first missing import, giving an exhaustive status report.
3. **Q: How does this contribute to pipeline reproducibility?**
   - **A:** Anyone replicating the study can run `python src/test_env.py` right after environment installation to certify their local environment is identical and ready.

---

## 6. `src/download_dataset.py`

### What it does
Streams the raw dataset archives (`dataset-original.zip` and `dataset-resized.zip`) directly from the Hugging Face repository into the local Hugging Face cache with resume support (`Range` headers) and progress telemetry.

### Why it exists
Prevents silent network hangs, rate-limit bottlenecks, or socket timeouts when fetching the large 3.63 GB high-resolution dataset archive, ensuring `load_dataset('garythung/trashnet')` can load cleanly and deterministically.

### Interview Questions & Answers
1. **Q: Why use HTTP `Range` headers in dataset acquisition?**
   - **A:** Large downloads over consumer networks can be interrupted by transient disconnects. Range headers allow resuming downloads from the exact byte offset where they stopped rather than starting from zero.
2. **Q: How does this script integrate with the Hugging Face `datasets` library?**
   - **A:** It places the downloaded archive into the exact cache blob path expected by `huggingface_hub` and links it into the repository snapshot, making `load_dataset('garythung/trashnet')` find it as a cache hit.
3. **Q: Why is telemetry (MB/s and ETA) essential in large ML data pipelines?**
   - **A:** Telemetry gives observability into bandwidth bottlenecks and enables early detection of stalled TCP sessions rather than allowing automated pipelines to block indefinitely.

---

## 7. `src/data_audit.py`

### What it does
Loads the full 5,054 examples from Hugging Face `garythung/trashnet`, extracts image dimensions, computes pixel MD5 and perceptual hashes (`pHash`, `dHash`, `aHash`), investigates the 2x count discrepancy against the GitHub repository (2,527 images), and generates `reports/dataset_metadata.csv`, `reports/class_counts_comparison.csv`, `reports/discrepancy_summary.json`, `reports/class_imbalance.png`, and `reports/sample_grid.png`.

### Why it exists
Enforces Phase 1 requirements: auditing the dataset, rigorously analyzing the 5,054 vs 2,527 mismatch in code without assumptions, mapping out class imbalances, and caching raw images locally for reproducible processing.

### Interview Questions & Answers
1. **Q: Why are pixel MD5 and perceptual hashes (pHash) both calculated?**
   - **A:** MD5 is a cryptographic hash sensitive to even 1-bit differences (detecting exact identical byte arrays), whereas perceptual hash (pHash) generates invariant frequency-based fingerprints that remain identical even when an image is resized, compressed, or slightly resampled.
2. **Q: What is the root cause of the 5,054 vs 2,527 image discrepancy?**
   - **A:** The Hugging Face repo contains both `dataset-original.zip` (2,527 full-res photos) and `dataset-resized.zip` (2,527 downscaled 512x384 photos). Hugging Face's auto-generated `imagefolder` loader ingested both archives into the single default `train` split, duplicating every sample at two resolutions (2,527 × 2 = 5,054).
3. **Q: Why save dataset metadata into CSV and JSON files in `reports/`?**
   - **A:** Adheres to Rule 2 ("NEVER invent numbers. Every number in README/REPORT must come from files your code saved in reports/"). It provides verifiable audit trails for recruitment judges.

---

## 8. `docs/DATASET.md`

### What it does
Provides the formal data specification for the EcoVision project, documenting the primary GitHub provenance (`6fa2b87`), the Hugging Face mirror layout, the exact class counts and imbalance percentages, the empirical resolution and hash investigation of the 2x mismatch, and cross-class duplicate findings with embedded visualization charts.

### Why it exists
Fulfills Phase 1 documentation deliverables and equips recruitment reviewers with a complete, verifiable reference explaining why the dataset has 5,054 rows on Hugging Face versus 2,527 on GitHub, and how class imbalance and label noise will be handled in subsequent phases.

### Interview Questions & Answers
1. **Q: Why is the 4.34:1 class imbalance between paper (23.51%) and trash (5.42%) problematic for evaluation?**
   - **A:** If evaluated using overall accuracy alone, an algorithm could predict the majority class and achieve ~75-80% accuracy while failing 100% of the time on miscellaneous trash. Macro-averaged metrics (macro-F1) weight all classes equally, exposing poor performance on minority classes.
2. **Q: How does the discovery of cross-class duplicates (e.g. `glass115.jpg` and `metal91.jpg`) affect model training?**
   - **A:** Cross-class duplicates represent conflicting label noise. If identical images are assigned contradictory labels, the loss function penalizes correct predictions and confuses gradients. In addition, if duplicates land in different splits, test set leakage occurs.
3. **Q: Why was `dataset-original.zip` paired with `dataset-resized.zip` on Hugging Face?**
   - **A:** The original TrashNet repository contained both raw high-resolution captures (up to 4000px) and a convenience downscaled set (`resize.py` output to 512×384). Hugging Face mirrored both archives without realizing its automatic dataset loader would combine both into the same split.

---

## 9. `src/dedupe_split.py`

### What it does
Restricts processing strictly to the 2,527 unique resized images, removes the 3 conflicting duplicate pairs (6 images) by filename logging them to `reports/removed_images.csv`, analyzes near-duplicate Hamming distance thresholds (0, 2, 4, 6), groups near-duplicates via connected components, executes a leak-proof Stratified GROUP Split (70% train, 15% val, 15% test, seed 42), and caches 224×224 images to `data/processed/`.

### Why it exists
Enforces Phase 2 requirements: grouping near-duplicate shots to reduce data contamination across splits (though near-duplicate leakage cannot be fully excluded), verifying minority class representation across all splits, and preprocessing images to the standard 224×224 input resolution for CNN training.

### Interview Questions & Answers
1. **Q: Why is Stratified GROUP Split necessary instead of standard stratified random splitting?**
   - **A:** If near-duplicate photos (e.g. the same object captured from two slightly different angles) are split randomly, one copy can end up in train and another in test. The model would memorize the specific object, yielding artificially inflated test metrics that fail on truly novel waste items.
2. **Q: Why remove conflicting duplicate pairs (`glass115`/`metal91`) completely instead of assigning one label?**
   - **A:** Without original experimental ground truth, any manual assignment introduces subjective bias. Removing both ambiguous instances (only 6 images out of 2,527) purges label noise cleanly without sacrificing statistical power.
3. **Q: Why pre-cache resized 224×224 images to disk rather than resizing on the fly in `Dataset.__getitem__`?**
   - **A:** On-the-fly resizing from 512×384 adds significant CPU overhead to every training epoch, causing GPU starvation on Windows with single-threaded data loading (`num_workers=0`). Pre-caching ensures ultra-fast I/O during model training.

---

## 10. `src/inspect_pairs.py`

### What it does
Identifies and visualizes the cross-class image pairs within perceptual hash (pHash) Hamming distance <= 4, saving them side-by-side to `reports/cross_class_pairs.png`, and extracts borderline near-duplicate pairs at Hamming distances 3 and 4 into `reports/near_duplicates_borderline.png`.

### Why it exists
Provides empirical verification of the chosen pHash threshold (<= 4). It confirms that the 2 cross-class pairs (`metal154` vs `plastic10`, `metal159` vs `plastic11`) are different physical objects (can vs. bottle) whose hashes overlap due to white posterboard backgrounds, justifying keeping both items in the dataset under the same group ID. It also demonstrates how the threshold handles borderline cases.

### Interview Questions & Answers
1. **Q: Why do metal cans and plastic bottles have low pHash distances despite being different materials?**
   - **A:** Perceptual hash computes a DCT on a downscaled 32x32 grayscale image and keeps only low-frequency components. A cylindrical soda can and a plastic water bottle centered horizontally on a pure white background share virtually identical low-frequency energy profiles, resulting in small Hamming distances (2 to 4).
2. **Q: Why keep these cross-class pairs instead of deleting them like the exact duplicates?**
   - **A:** The 3 removed pairs were identical pixel photos labeled with two contradictory classes (genuine labeling errors). In contrast, `metal159` and `plastic11` are physically distinct waste items with correct labels. Deleting them would throw away valid data simply because low-frequency hashing cannot differentiate cylindrical shapes. Grouping them preserves data integrity and prevents train/val/test leakage.
3. **Q: Why inspect borderline pairs at distance 3 and 4 rather than only distance 0?**
   - **A:** Distance 0 only shows exact or near-identical images. The hardest cases and highest risk of false merges or missed duplicates occur at the threshold boundary (3 and 4). Visual inspection of the borderline pairs proves that threshold 4 captures meaningful similarities without collapsing unrelated images into mega-clusters.

---

## 11. `src/models.py`

### What it does
Defines the PyTorch neural network architectures for EcoVision:
1. `BaselineCNN`: A small 4-block convolutional neural network built from scratch (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d), followed by Global Average Pooling, Dropout (0.3), and a 6-class linear classification head (~390k parameters).
2. `create_mobilenetv2_transfer`: Builder function that loads ImageNet-pretrained `MobileNetV2` from `torchvision.models` and adapts it for EcoVision by replacing the default 1000-class classifier head with `Dropout(p=0.3)` and `Linear(1280, 6)`. Total parameters: ~2.23M.

### Why it exists
Centralizes model definitions across phases into a single, clean module. It ensures model instantiation is strictly reproducible, modular, and directly importable by training pipelines and deployment interfaces alike.

### Interview Questions & Answers
1. **Q: What is an Inverted Residual Block with Linear Bottleneck in MobileNetV2?**
   - **A:** Unlike standard ResNet blocks that compress channels then expand them (narrow-wide-narrow), MobileNetV2 expands channels using 1x1 convolutions into high-dimensional space, applies a lightweight 3x3 depthwise separable convolution, and projects back down to a low-dimensional bottleneck without a non-linear activation (linear bottleneck). This preserves information manifold integrity while drastically minimizing compute and memory footprint.
2. **Q: Why choose MobileNetV2 over heavier backbones like ResNet50 or ViT for this recruitment task?**
   - **A:** TrashNet has only ~1,763 training images. Heavy models with 25M+ parameters risk severe overfitting, high latency, and heavy VRAM usage. MobileNetV2 has only ~2.2M parameters and low latency, making it ideal for edge deployment (e.g. smart recycling bins) while offering rich ImageNet semantic representations.
3. **Q: Why use Global Average Pooling (GAP) before the classifier?**
   - **A:** GAP converts each spatial feature map ($H \times W$) into a single scalar average. This enforces translation invariance, prevents spatial overfitting, and avoids the parameter explosion of large dense fully connected layers.

---

## 12. `src/train_baseline.py`

### What it does
Executes the training and validation pipeline for `BaselineCNN`. It loads 224x224 images from `data/processed/{train,val}`, applies ImageNet normalization without augmentation, computes balanced class weights strictly from the train split, optimizes with Adam (`lr=1e-3`), applies early stopping on validation macro-F1 (patience 7), checkpoints the best weights to `models/baseline_cnn.pt`, logs training history to `reports/baseline_history.csv`, generates loss/macro-F1 curves in `reports/baseline_curves.png`, and outputs detailed validation metrics to `reports/baseline_val_metrics.json`.

### Why it exists
Fulfills Phase 3 modeling requirements while strictly adhering to core ML principles: evaluating only on validation data and keeping the test set completely pristine for final evaluation.

### Interview Questions & Answers
1. **Q: Why calculate class weights strictly from the training split?**
   - **A:** Deriving class weights from validation or test splits introduces data leakage (specifically, prior distribution leakage). The model and loss function must only access information available during training; the validation split must remain an uninfluenced proxy for unseen data.
2. **Q: Why use macro-F1 for early stopping rather than validation loss or accuracy?**
   - **A:** TrashNet exhibits a 4.34:1 class imbalance, where the `trash` category represents only 5.4% of samples. A model could achieve high accuracy or lower overall loss while predicting poorly on minority classes. Macro-F1 weights all classes equally, stopping training when performance across all categories—especially rare classes—is maximized.
3. **Q: Why is data augmentation explicitly disabled in this phase?**
   - **A:** Scientific rigor requires isolating experimental variables. Disabling augmentation in Phase 3 establishes the true baseline performance on the raw processed data. When augmentation is evaluated in Phase 5, any improvement can be directly attributed to augmentation techniques rather than confounded model changes.

---

## 13. `src/train_transfer.py`

### What it does
Executes the two-stage transfer learning pipeline with ImageNet-pretrained MobileNetV2:
- **Stage 1 (Feature Extraction):** Completely freezes the backbone (`requires_grad=False`), training only the new 6-class classification head with Adam (`lr=1e-3`, max 10 epochs, patience 3 on validation macro-F1).
- **Stage 2 (Fine-Tuning):** Loads best Stage 1 weights, unfreezes the last 3 inverted-residual blocks of the backbone (`features[-3:]`), keeps BatchNorm layers of all frozen blocks in `eval()` mode, and fine-tunes using discriminative learning rates (`1e-4` for backbone blocks, `1e-3` for head) with Adam (max 20 epochs, patience 5 on validation macro-F1).
Saves the best checkpoint to `models/mobilenetv2_transfer.pt`, logs per-stage history to `reports/transfer_history.csv`, plots transition curves to `reports/transfer_curves.png`, generates side-by-side validation confusion matrices in `reports/val_confusion_matrices.png`, and outputs a comparative benchmark against the baseline in `reports/baseline_vs_transfer_val.csv`.

### Why it exists
Fulfills Phase 4 transfer learning requirements, rigorously evaluating whether pre-trained visual representations from ImageNet solve waste material classification significantly better than learning from scratch, while demonstrating proper two-stage fine-tuning discipline.

### Interview Questions & Answers
1. **Q: What is transfer learning and why is it effective for small datasets like TrashNet?**
   - **A:** Transfer learning leverages representations learned by a neural network trained on a massive source dataset (like ImageNet with 1.2M images) and applies them to a different target task. Low- and mid-level convolutional features (edges, textures, surfaces, geometric contours) are largely universal across natural images. Reusing these pre-learned filters prevents overfitting on small datasets where training deep filters from scratch is statistically constrained.
2. **Q: Why freeze the backbone in Stage 1 before fine-tuning in Stage 2?**
   - **A:** The newly initialized classification head has random weights. If the entire network were trained immediately with high learning rates, massive gradient updates from the random head would backpropagate through the backbone and destroy ("catastrophically overwrite") the carefully learned pretrained feature detectors. Freezing the backbone allows the head to align with pretrained features first; only then is gentle fine-tuning applied.
3. **Q: Why must BatchNorm layers in frozen blocks remain in `eval()` mode during fine-tuning?**
   - **A:** If a BatchNorm layer is left in `train()` mode, it continues to compute running mean and variance from incoming mini-batches. When training on a small dataset with small batch sizes (32), noisy mini-batch statistics corrupt the stable ImageNet statistics embedded in the frozen weights, leading to severe distribution shift and degradation of upstream feature representations.

---

## 14. `src/train_augmentation.py`

### What it does
Executes the Phase 5 data augmentation comparison experiment for MobileNetV2 on TrashNet. It compares an un-augmented baseline against an augmented configuration across 3 random seeds (42, 43, 44), strictly on the validation set without touching the test set. It trains models using the identical two-stage transfer learning procedure (head training followed by discriminative fine-tuning), tracks validation macro-F1, logs detailed per-run metrics to `reports/augmentation_runs.csv`, calculates summary statistics (mean ± std) in `reports/augmentation_summary.csv`, generates visualizations (`reports/augmentation_effect.png` and `reports/augmentation_gap.png`), selects the best-performing model checkpoint based on validation macro-F1, and saves it to `models/mobilenetv2_final.pt`.

### Why it exists
Fulfills Phase 5 experimental requirements by empirically measuring the performance delta, error bars, and generalization loss gap resulting from data augmentation. It establishes a rigorous scientific framework to justify whether data augmentation should be included in the final model pipeline.

### Interview Questions & Answers
1. **Q: Why run the experiment across 3 random seeds instead of just one?**
   - **A:** Deep learning optimization is inherently stochastic due to random weight initialization, mini-batch sequencing, and non-deterministic CUDA operations. A single run might show an apparent improvement purely due to favorable seed luck. Evaluating across multiple seeds (42, 43, 44) allows calculating mean and standard deviation, verifying that the performance gain (+0.0179 macro-F1) is consistent across 3 seeds and that every augmented seed strictly outperforms every non-augmented seed (though 3 seeds is a small sample).
2. **Q: Why keep the test split strictly untouched during the augmentation experiment?**
   - **A:** The test set serves as the final, unbiased holdout dataset used to estimate real-world generalization in Phase 6. If the test set were evaluated during hyperparameter tuning, model comparison, or seed selection, information would leak into the experimental decision process, invalidating the statistical independence and integrity of the final benchmark.
3. **Q: How does `train_augmentation.py` measure and report the train/validation loss gap?**
   - **A:** It records the training loss and validation loss at the checkpoint epoch corresponding to the best validation macro-F1. The difference ($\text{Val Loss} - \text{Train Loss}$) quantifies the degree of overfitting. By comparing this gap between configurations (0.2769 for non-augmented vs 0.1918 for augmented), the script empirically proves that data augmentation acts as an effective regularizer, mitigating memorization of training instances.

---

## 15. `docs/AUGMENTATION.md`

### What it does
Documents the full experimental specification, empirical findings, and model selection decision for Phase 5. It describes the exact augmentation transforms applied, details the rationale for augmenting only the training set, presents complete mean ± std summary and per-seed results tables, analyzes the impact on macro-F1, loss gap, training time, and per-class performance, announces the selection of MobileNetV2 Augmentation Seed 44 as the final model (`mobilenetv2_final.pt`), and outlines honest scientific caveats (sample size, single validation split, minority class volatility, and validation selection bias).

### Why it exists
Serves as the authoritative, permanent record of the Phase 5 experiment. It adheres to Rule 2 ("NEVER invent numbers. Every number in README/REPORT must come from files your code saved in reports/") by pulling all figures directly from saved CSV reports, providing complete transparency and auditability for recruiters and technical reviewers.

### Interview Questions & Answers
1. **Q: Why apply data augmentation strictly to the training set and never to the validation or test sets?**
   - **A:** The validation and test sets must accurately reflect the uncorrupted, real-world data distribution that the model will encounter in deployment. Augmenting evaluation sets introduces synthetic artifacts (e.g. artificial color shifts, aggressive crops) and random variance into the benchmarks, corrupting evaluation integrity. Augmentation is an optimization-time regularization technique designed to force the network to learn invariant representations from diverse training views.
2. **Q: Why did the minority `trash` class experience the largest F1 improvement (+0.0646) under augmentation?**
   - **A:** With only 137 total unique images in TrashNet (and ~96 in train), the model has very few physical examples to learn the wide intra-class diversity of miscellaneous waste. Data augmentation (rotations, flips, color jitter, and random resized crops) synthesizes novel perspectives and lighting variations, drastically enriching the effective sample density for minority classes where feature sparsity is most severe.
3. **Q: Why is selecting the best model seed based on validation performance slightly optimistic, and how is this addressed?**
   - **A:** Choosing the top-performing seed on validation (Seed 44 with macro-F1 = 0.9221) introduces a mild selection bias known as the "winner's curse," because the selected model may have benefited from favorable sample alignment on that specific validation set. This caveat is acknowledged openly in scientific reporting, and the holdout test set in Phase 6 serves as the definitive, unbiased arbiter of true model performance.

---

## 16. `src/evaluate_test.py`

### What it does
Executes the final holdout evaluation for Phase 6. It touches the test split (379 images) for the first and only time, loading `models/mobilenetv2_final.pt` and `models/baseline_cnn.pt` without augmentation. It logs per-image prediction probabilities to `reports/val_predictions.csv` and `reports/test_predictions.csv`, computes comprehensive classification metrics in `reports/test_metrics.json` and `reports/test_comparison.csv`, calculates 1,000-resample non-parametric bootstrap 95% confidence intervals in `reports/test_bootstrap_ci.json`, generates raw and row-normalized confusion matrices in `reports/test_confusion_matrix.png`, exports all misclassified test samples to `reports/test_errors.csv`, and visualizes failure cases in `reports/failure_cases.png`.

### Why it exists
Provides the definitive, unbiased benchmark of real-world model generalization. It proves that the model's high performance is not an artifact of validation overfitting, measures the true empirical margin over the scratch baseline (+21.90% accuracy, +0.2170 macro-F1), and surfaces failure modes for post-deployment monitoring.

### Interview Questions & Answers
1. **Q: Why must the holdout test set only be evaluated once after all training and model decisions are complete?**
   - **A:** Evaluating repeatedly on a test set and tweaking hyperparameters, architectures, or augmentations based on test feedback causes data leakage ("data dredging" or test-set overfitting). The holdout test set must remain completely isolated from the model selection feedback loop so that its final score serves as an honest, statistically valid proxy for true out-of-distribution performance.
2. **Q: Why use non-parametric bootstrapping (1,000 resamples) to compute confidence intervals?**
   - **A:** Point estimates (such as "89.71% accuracy") conceal the sampling variability inherent to finite test datasets ($N=379$). Bootstrapping samples with replacement 1,000 times to construct empirical sampling distributions for accuracy and macro-F1 without assuming asymptotic normality, yielding robust 95% confidence intervals (accuracy: [0.8654, 0.9261], macro-F1: [0.8195, 0.9036]).
3. **Q: Why did test macro-F1 (0.8647) decline relative to validation macro-F1 (0.9221) while overall test accuracy remained high (89.71%)?**
   - **A:** Macro-F1 weights each class equally regardless of support. In the holdout test set, the severe minority `trash` class contains only 21 images and achieved an F1 of 0.6316 (due to 9 false negatives), pulling down the unweighted macro mean. Meanwhile, overall accuracy is dominated by well-represented classes like cardboard (F1 0.9580, support 60) and paper (F1 0.9438, support 89), where performance was outstanding.

---

## 17. `src/tune_threshold.py`

### What it does
Executes Phase 7 confidence threshold optimization using strictly `reports/val_predictions.csv`. It sweeps confidence cutoffs from 0.30 to 0.95 in increments of 0.05, designating predictions below the threshold as "uncertain". For each threshold, it calculates coverage, accuracy, and macro-F1 on answered items, logs metrics to `reports/threshold_sweep.csv`, and visualizes trade-off curves in `reports/threshold_curve.png`. It selects the operational threshold using a pre-committed rule (highest coverage with validation accuracy $\ge$ 0.95), stores the decision in `models/threshold.json`, and applies the chosen threshold once to `reports/test_predictions.csv` (logging results to `reports/test_with_threshold.json`).

### Why it exists
Enables selective classification (prediction with a reject option) for automated deployment. In real-world waste processing, false positive classifications can contaminate an entire bale of recyclable material. Rejection of low-confidence predictions allows the system to operate at high precision on automated items while safely routing ambiguous items to secondary sorting.

### Interview Questions & Answers
1. **Q: Why tune the confidence threshold strictly on validation predictions rather than test predictions?**
   - **A:** Tuning on test predictions would contaminate the holdout split, yielding an overfitted threshold that exploits idiosyncratic test sample noise. The threshold must be learned exclusively from validation data; the test set is only used once to measure how well that threshold generalizes to unseen operational data.
2. **Q: What is the operational trade-off between coverage and accuracy in selective classification?**
   - **A:** Raising the threshold eliminates low-confidence predictions, which reduces false positives and boosts accuracy on the remaining items, but simultaneously lowers coverage (fewer items handled automatically). The engineering objective is to identify the Pareto-optimal operating point that satisfies safety/quality constraints (e.g. $\ge$ 95% accuracy) while maximizing automated throughput.
3. **Q: Why did threshold 0.80 achieve 95.14% test accuracy while rejecting only 13.19% of samples?**
   - **A:** The model's misclassifications disproportionately cluster at lower confidence levels (under 0.80). Setting the rejection cutoff at 0.80 removed 23 of the 39 test errors (while rejecting 27 correct predictions), concentrating errors into the "uncertain" bucket and increasing precision for the 86.81% of items that were confidently accepted.

---

## 18. `docs/TEST_RESULTS.md`

### What it does
Synthesizes the complete empirical results from Phase 6 holdout test evaluation and Phase 7 confidence thresholding. It presents side-by-side benchmark tables comparing Baseline CNN against MobileNetV2 Final with 95% bootstrap confidence intervals, documents the validation threshold sweep and test deployment metrics, provides detailed per-class error analysis, and outlines honest scientific limitations (small test set, extreme minority class volatility, studio photography bias, and softmax overconfidence).

### Why it exists
Serves as the definitive, auditable technical report for engineering reviews and hiring managers. It adheres strictly to Rule 2 ("NEVER invent numbers") by extracting every figure from saved CSV and JSON reports, openly disclosing that test performance is lower than validation and explaining why.

### Interview Questions & Answers
1. **Q: Why is it critical to report that test macro-F1 (0.8647) is lower than validation macro-F1 (0.9221)?**
   - **A:** Honesty and scientific integrity are fundamental in applied machine learning. Model selection on validation data inherently produces a slightly optimistic estimate ("winner's curse"). Acknowledging this performance gap openly builds trust and provides realistic performance expectations for deployment.
2. **Q: Why do glass and plastic represent the primary cross-class confusion pair in TrashNet?**
   - **A:** Both materials frequently appear as transparent, cylindrical bottles or containers photographed against plain white backgrounds. Their low- and mid-level visual features—such as specular highlights, transparency, and geometric contours—are nearly indistinguishable without material-specific spectral or physical signals.
3. **Q: How does dataset collection methodology (clean white posterboard backgrounds) impact real-world generalizability?**
   - **A:** TrashNet images were captured under controlled laboratory conditions with centered single objects. In real-world recycling facilities or consumer smart bins, objects are crushed, soiled, overlapping, and viewed under varying ambient lighting against cluttered backgrounds. A model trained only on clean studio images experiences domain shift when deployed in the wild unless augmented with in-situ data.

---

## 19. `src/predict.py`

### What it does
Implements the inference module for EcoVision with both a CLI (`python src/predict.py path/to/image.jpg`) and a programmatic API (`predict_image(pil_image)`). It mirrors the exact preprocessing pipeline from dataset caching (RGB conversion, direct 224x224 Bilinear resizing without center-cropping, ToTensor, and ImageNet standardization). It loads `models/mobilenetv2_final.pt`, reads the optimized confidence threshold from `models/threshold.json` (0.80), marks predictions below the threshold as UNCERTAIN, and returns full class probability distributions, confidence scores, and top-3 rankings. It also ensures `models/classes.json` is exported.

### Why it exists
Serves as the core inference engine bridging model training to downstream consumer applications (CLI utilities, API endpoints, and web demo interfaces). It guarantees preprocessing consistency, preventing silent train-serve skew.

### Interview Questions & Answers
1. **Q: Why is matching the exact preprocessing pipeline (direct Bilinear resize without center crop) critical for inference?**
   - **A:** Train-serve skew is one of the most common silent failure modes in computer vision systems. If training was conducted on directly resized 224x224 images, introducing an un-trained transformation like center-cropping at inference time would crop out peripheral features (e.g. bottle caps, can lids, box edges) that the network relied on, degrading inference accuracy.
2. **Q: Why implement caching for model loading (`get_model`) in Python inference code?**
   - **A:** Instantiating deep learning models and loading multi-megabyte weight tensors into memory or GPU VRAM incurs substantial latency overhead (100–500ms). Caching the model instance globally allows subsequent inference calls to execute in single-digit milliseconds without reloading weights from disk.
3. **Q: How does the module handle predictions when confidence falls below the threshold?**
   - **A:** The module flags the output as `uncertain: True` while still reporting the top predicted category and full class probability distribution. This design allows downstream consumers (like smart sorting bins) to safely divert ambiguous items to manual or secondary sorting bins rather than committing an automated sorting error.

---

## 20. `src/measure_latency.py`

### What it does
Benchmarks model inference latency on both CPU and GPU hardware across 10 warmup and 100 timed iterations. It measures median and 95th percentile (p95) latency for both model-only forward passes and end-to-end processing (image opening, resizing, normalization, and forward pass). It computes model file size (8.75 MB) and parameter count (2,231,558), logging complete benchmark telemetry to `reports/latency.json`.

### Why it exists
Validates edge deployment feasibility. In practical automated waste sorting applications, items move past cameras at speeds requiring low latency (e.g. <50ms per item). This script proves that MobileNetV2 satisfies low-latency sorting constraints on both GPU (~12ms e2e) and commodity CPU (~24ms e2e).

### Interview Questions & Answers
1. **Q: Why is `torch.cuda.synchronize()` strictly necessary when measuring GPU latency?**
   - **A:** PyTorch executes CUDA kernel launches asynchronously on the GPU while the CPU continues executing Python instructions concurrently. If `torch.cuda.synchronize()` is not called before and after timing blocks, Python's wall-clock timer will measure only the negligible CPU kernel dispatch time rather than the actual GPU kernel computation time.
2. **Q: Why measure both model-only and end-to-end latency?**
   - **A:** Model-only latency measures theoretical forward-pass compute efficiency, but real-world deployed systems must read image files from sensors/disk, decode JPEG bytes, resize pixels, and convert formats. End-to-end measurement captures I/O and CPU preprocessing overhead, giving an honest appraisal of actual system throughput.
3. **Q: Why use median and p95 rather than only mean latency?**
   - **A:** Mean latency is sensitive to transient system spikes (OS context switches, garbage collection pauses, thermal throttling). The median represents typical throughput, while p95 captures tail latency, which dictates service-level agreements (SLAs) in real-time embedded systems.

---

## 21. `app.py`

### What it does
Launches the local Gradio web application for EcoVision. Accepts image uploads (`type="pil"`), feeds them through `predict_image`, and displays a `gr.Label` component with all 6 class probabilities alongside a status text line indicating whether the sample is accepted (`"Predicted: <class> (confidence XX.X%)"`) or rejected as uncertain (`"Uncertain: confidence below threshold, closest class: <class>"`). Registers 6 test split examples under `demo_examples/` and provides honest documentation regarding studio photography domain shift.

### Why it exists
Fulfills Phase 9 interactive demo requirements, providing stakeholders, end-users, and technical interviewers with an intuitive graphical interface to test model predictions and explore threshold behavior interactively.

### Interview Questions & Answers
1. **Q: Why use Gradio's `gr.Label` with top-6 classes rather than simply showing the top-1 prediction text?**
   - **A:** `gr.Label` visualizes the entire softmax posterior probability distribution across all 6 classes. This gives transparency into model certainty and reveals secondary candidate classes (e.g., when the model is torn between glass and plastic), aiding user trust and diagnostic error analysis.
2. **Q: Why include pre-loaded test examples in the UI?**
   - **A:** Providing curated test set samples ensures immediate zero-friction evaluation for reviewers without requiring them to search for, download, or transfer test images to the machine.
3. **Q: Why explicitly include the disclaimer that real campus waste may be harder?**
   - **A:** Scientific integrity requires communicating model limitations. TrashNet contains clean single-item studio photos on plain posterboard. Real-world waste items exhibit heavy deformation, staining, occlusions, and varied backgrounds, so transparently stating domain shift sets appropriate operational expectations.

---

## 22. `space/app.py`

### What it does
Provides a completely self-contained, CPU-safe Gradio application tailored for deployment on Hugging Face Spaces. It imports zero code from `src/`, loads all assets locally (`model.pt`, `classes.json`, `threshold.json`, `demo_examples/`), executes exclusively on CPU (`map_location="cpu"`), mirrors the exact training preprocessing pipeline, and exposes the identical interactive interface.

### Why it exists
Hugging Face Spaces containers often run in isolated Linux CPU environments without access to local repository packages or GPU acceleration. Isolating the deployment package into a standalone directory ensures that the public cloud demo remains robust, zero-dependency, and immediately deployable.

### Interview Questions & Answers
1. **Q: Why must `space/app.py` not import anything from `src/`?**
   - **A:** When deployed to Hugging Face Spaces, only the contents of the `space/` directory (or Git repository root of the Space) are deployed into the cloud container. If `app.py` attempted to import from `src.predict`, the container would immediately crash with `ModuleNotFoundError`.
2. **Q: Why enforce `map_location="cpu"` and strictly CPU tensors?**
   - **A:** Free Hugging Face Spaces provide 2 vCPU hardware tiers without dedicated GPUs. Attempting to deserialize CUDA tensors or call `.cuda()` without an active GPU causes fatal PyTorch runtime exceptions.
3. **Q: How does the standalone architecture definition prevent version incompatibility?**
   - **A:** `space/app.py` directly defines `build_model` by wrapping `torchvision.models.mobilenet_v2(weights=None)` and swapping the classifier head. Because it loads a plain `state_dict` rather than a pickled model object, it avoids class serialization path mismatches between development environments and deployment containers.

---

## 23. `space/requirements.txt`

### What it does
Specifies the exact minimal CPU-only wheel dependencies for the Hugging Face Space. Its first line instructs pip to use PyTorch's official CPU wheel index (`--extra-index-url https://download.pytorch.org/whl/cpu`), followed by pinned versions `torch==2.6.0`, `torchvision==0.21.0`, `pillow`, and `numpy`.

### Why it exists
Standard PyPI wheels for PyTorch include bundled NVIDIA CUDA runtime libraries that bloat the container installation by 2–3 GB, causing slow builds and exceeding disk limits. The CPU index installs lightweight wheels (~150 MB) while deliberately excluding `gradio` (which Hugging Face Spaces injects and manages automatically).

### Interview Questions & Answers
1. **Q: Why should `gradio` be omitted from `space/requirements.txt`?**
   - **A:** Hugging Face Spaces automatically reads the Gradio version from the `README.md` YAML metadata block (`sdk_version: 6.30.0`) and pre-installs it in the base image. Declaring `gradio` in `requirements.txt` can cause conflicting dependency resolutions or redundant installation overhead.
2. **Q: What is the benefit of pointing `--extra-index-url` to PyTorch's CPU wheel repository?**
   - **A:** It downloads lean CPU-only binaries rather than bloated CUDA-enabled wheels, speeding up Hugging Face container build times from several minutes to seconds and conserving container ephemeral storage.
3. **Q: Why pin `torch==2.6.0` and `torchvision==0.21.0`?**
   - **A:** Pinning the exact minor versions prevents breaking changes in torch/torchvision APIs, ensuring that model weight deserialization and torchvision model architecture builders remain 100% deterministic.

---

## 24. `space/README.md`

### What it does
Serves as the configuration manifest and documentation landing page for the Hugging Face Space. It defines YAML front matter (`title: EcoVision`, `emoji: ♻️`, `colorFrom: green`, `colorTo: blue`, `sdk: gradio`, `sdk_version: 6.30.0`, `app_file: app.py`, `pinned: false`) followed by a concise 5-line project overview and honest capability description.

### Why it exists
Hugging Face Spaces parser relies on the YAML header at the top of `README.md` to configure the Space runtime environment (selecting the Gradio SDK version, launch file, and visual card styling).

### Interview Questions & Answers
1. **Q: How does Hugging Face Spaces use the YAML front matter in `README.md`?**
   - **A:** Hugging Face's container builder parses the YAML front matter before building the image. It uses `sdk: gradio` and `sdk_version` to bootstrap the container with the correct web framework runtime and executes `app_file` upon container start.
2. **Q: Why is pinning `sdk_version` critical in Spaces configuration?**
   - **A:** If `sdk_version` is omitted or unpinned, Hugging Face defaults to the latest Gradio release, which may introduce breaking API changes (such as changes to `gr.Interface`, components, or flagging options), causing the deployment to crash unexpectedly.
3. **Q: What is the role of the 5-line text description below the YAML block?**
   - **A:** It provides immediate context to visitors, explaining the model architecture, supported waste categories, selective thresholding behavior, and transparent disclosures about domain shift from studio photos.

---

## 25. `docs/architecture.png` & `docs/architecture.mmd`

### What it does
Visualizes the end-to-end data processing, model training, validation selection, threshold tuning, test auditing, and edge deployment pipeline for EcoVision. `docs/architecture.mmd` provides the version-controlled Mermaid diagram source, while `docs/architecture.png` is a high-resolution (220 DPI) graphical representation generated using `matplotlib` with zero overlapping text boxes or crossing flow lines.

### Why it exists
Provides immediate architectural clarity for engineering reviews, recruitment audits, and documentation landing pages, showing how raw dataset archives are transformed into leak-proof splits and served on edge hardware.

### Interview Questions & Answers
1. **Q: Why maintain both Mermaid code (`.mmd`) and a rendered PNG (`.png`)?**
   - **A:** Mermaid provides a text-based, version-controllable specification that can be edited and reviewed via Git diffs, while the static PNG guarantees identical, high-fidelity rendering across all markdown viewers, offline readers, and platforms that do not support dynamic Mermaid parsing.
2. **Q: How does the diagram reflect the strict separation of validation and test stages?**
   - **A:** The diagram visually routes model selection (Augmentation Seed 44) and confidence threshold optimization (0.80 cutoff) strictly from the validation split into the final model checkpoint, while the holdout test set is connected via a single, unidirectional arrow evaluated only once in Phase 6.
3. **Q: Why are both model-only and end-to-end latency benchmarks illustrated on the architecture card?**
   - **A:** It demonstrates full-system engineering awareness. Neural compute latency (9.22 ms on GPU) only reflects tensor arithmetic, whereas end-to-end latency (11.94 ms on GPU / 24.38 ms on CPU) accounts for image decoding, memory copies, and preprocessing, proving real-world edge feasibility.

---

## 26. `README.md`

### What it does
Serves as the primary documentation portal and entry point for the entire EcoVision repository. It provides an executive summary, benchmark results tables (including validation scores and test 95% bootstrap confidence intervals), dataset provenance and 2x mirror discrepancy explanations, class imbalance statistics, the architecture diagram, two-stage fine-tuning methodology, reproduce-from-scratch instructions, CLI and API usage examples, latency benchmarks, confidence threshold analysis, failure case visualizations, honest technical caveats, and formal dataset attribution.

### Why it exists
Fulfills Phase 10 top-level documentation deliverables. It adheres strictly to the rule that every reported number must originate from files in `reports/` or `models/`, ensuring complete auditability and credibility for technical reviewers and hiring managers.

### Interview Questions & Answers
1. **Q: Why include both point estimates and 95% bootstrap confidence intervals in the main README benchmark table?**
   - **A:** Point estimates (like 89.71% test accuracy) hide finite-sample variance on a 379-image test split. Reporting 95% bootstrap confidence intervals ([86.54%, 92.61%] for accuracy; [0.8195, 0.9036] for macro-F1) provides rigorous scientific transparency regarding the true underlying generalization bounds.
2. **Q: How are public links configured in the README badges?**
   - **A:** They link directly to the public GitHub repository and the live Streamlit demo URL (`https://ecovision-evzcs2cqdcwsrhk3fvdbxu.streamlit.app`), providing one-click access for reviewers and recruitment evaluators.
3. **Q: Why does the README openly report that the test set rejected more images than the validation set under threshold 0.80?**
   - **A:** Scientific integrity requires communicating empirical findings honestly. Test data contains natural visual diversity and edge cases not present during validation, causing 13.19% of test samples (50 images) to fall below the cutoff compared to 7.12% on validation (27 images), while maintaining high precision (95.14% accuracy on answered items).

---

## 27. `REPORT.md`

### What it does
Provides a formal, comprehensive 2-page technical evaluation and audit report summarizing the empirical findings of the EcoVision project. It details dataset provenance and cleaning, class distribution, baseline vs. transfer learning progression, the 3-seed augmentation comparison, holdout test metrics with bootstrap confidence intervals, selective classification coverage vs. accuracy trade-offs, visual error analysis, hardware latency benchmarks, real-world campus waste limitations, and strategic recommendations for real-world deployment.

### Why it exists
Equips engineering leads, auditors, and recruitment evaluators with a thorough, publication-grade technical report that documents the scientific rigor, statistical significance, and engineering trade-offs behind every decision in the project.

### Interview Questions & Answers
1. **Q: What is the primary purpose of `REPORT.md` relative to `README.md`?**
   - **A:** While `README.md` is an actionable developer onboarding guide (with setup commands, CLI snippets, and architecture overviews), `REPORT.md` is a formal technical whitepaper focusing on experimental methodology, statistical significance, error taxonomy, and operational risk analysis.
2. **Q: How does `REPORT.md` justify the selection of MobileNetV2 over the scratch baseline?**
   - **A:** It presents side-by-side empirical metrics showing a massive +21.90% accuracy gain (89.71% vs. 67.81%) and +0.2170 macro-F1 gain (0.8647 vs. 0.6477), demonstrating that ImageNet pre-trained feature hierarchies are essential for generalizing from a small 1,763-image training split.
3. **Q: What deployment recommendation does the report provide regarding minority class performance?**
   - **A:** The report recommends targeting active data collection specifically for miscellaneous `trash` (which only had 21 test samples and an F1 of 0.6316) and expanding beyond single-item classification toward object detection for multi-item waste streams.

---

## 28. `docs/INTERVIEW_QA.md`

### What it does
Compiles 30 comprehensive, plain-language technical interview questions and answers grounded directly in EcoVision's empirical numbers, architecture, and code. Topics span CNN fundamentals, transfer learning, MobileNetV2 architecture, two-stage fine-tuning, train/val/test split hygiene, near-duplicate leakage prevention, class imbalance and inverse frequency weighting, macro-F1 vs. accuracy, confusion matrix interpretation, overfitting and loss gap analysis, the 3-seed augmentation experiment, early stopping, selective classification thresholding, bootstrap confidence intervals, GPU latency synchronization, train-serve skew elimination, API contracts, Git hygiene, and real-world campus waste limitations.

### Why it exists
Serves as an exhaustive technical preparation guide for code reviews and recruitment interviews, demonstrating deep theoretical mastery paired with authentic hands-on empirical experience.

### Interview Questions & Answers
1. **Q: Why are real numbers from `reports/` referenced in nearly every answer in `INTERVIEW_QA.md`?**
   - **A:** Quoting verified empirical figures (such as "390,438 baseline parameters", "+0.0179 macro-F1 augmentation gain", "0.80 cutoff achieving 95.14% test accuracy", and "11.94 ms GPU latency") proves that the developer directly built, measured, and debugged the system rather than reciting generic textbook definitions.
2. **Q: How does `INTERVIEW_QA.md` explain the trade-off between coverage and accuracy in selective classification?**
   - **A:** It explains that raising the confidence threshold filters out low-confidence, error-prone predictions, increasing accuracy on accepted items but reducing the proportion of items automated. A cutoff of 0.80 achieved the Pareto-optimal operating target ($\ge 95\%$ accuracy on answered items) while automating 86.81% of test waste.
3. **Q: How does the document address the question of real-world deployment limitations?**
   - **A:** It honestly acknowledges domain shift: TrashNet consists of clean, single items on white posterboard, whereas real campus waste is crushed, food-soiled, multi-object, and viewed under variable lighting. Deploying to real bins requires in-situ collection and multi-label object detection.

---

## 29. `docs/REQUIREMENTS_CHECKLIST.md`

### What it does
Provides a comprehensive traceability matrix mapping every foundational engineering requirement, modeling objective, brownie point specification, and deliverable across all 11 project phases directly to the files, code modules, and test reports that prove their completion. Each item is audited with a status badge (`Done` / `Partial` / `Not done`) alongside rigorous technical verification notes.

### Why it exists
Fulfills Phase 11 audit requirements by establishing an unambiguous, transparent verification contract. It allows evaluators, hiring managers, and code auditors to cross-examine every claimed feature against concrete repository artifacts without guessing or manual searching.

### Interview Questions & Answers
1. **Q: Why is a requirements traceability matrix valuable in commercial machine learning projects?**
   - **A:** ML projects often suffer from scope creep or incomplete validation where models are trained but critical operational safeguards (such as leak-proof splitting, confidence rejection, or edge latency profiling) are skipped. A traceability matrix guarantees that every business and technical constraint has been implemented and empirically proven.
2. **Q: How does the checklist prove that data leakage was prevented?**
   - **A:** It references `src/dedupe_split.py` and `reports/near_duplicates_metrics.json`, explaining how connected components on perceptual hash Hamming distances $\le 4$ ensured that identical object poses were grouped atomically into the same split rather than leaking between train and test.
3. **Q: What role do the "technical verification notes" play in recruitment reviews?**
   - **A:** Rather than just checking off a box, the verification notes explain the exact algorithmic mechanism (e.g. inverse-frequency class weights, two-stage fine-tuning with frozen BatchNorm, 1,000-resample bootstrap CIs), demonstrating that the developer understands the underlying engineering rationale.





