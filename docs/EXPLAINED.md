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
3. **Q: Why is telemetry (MB/s and ETA) essential in production ML data pipelines?**
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
Enforces Phase 2 requirements: preventing data contamination across splits, ensuring zero leakage of near-duplicate shots, verifying minority class representation across all splits, and preprocessing images to the standard 224×224 input resolution for CNN training.

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
Fulfills Phase 5 experimental requirements by empirically measuring the performance delta, error bars, and generalization loss gap resulting from data augmentation. It establishes a rigorous scientific framework to justify whether data augmentation should be included in the final production pipeline.

### Interview Questions & Answers
1. **Q: Why run the experiment across 3 random seeds instead of just one?**
   - **A:** Deep learning optimization is inherently stochastic due to random weight initialization, mini-batch sequencing, and non-deterministic CUDA operations. A single run might show an apparent improvement purely due to favorable seed luck. Evaluating across multiple seeds (42, 43, 44) allows calculating mean and standard deviation, verifying that the performance gain (+0.0179 macro-F1) is statistically robust and that every augmented seed strictly outperforms every non-augmented seed.
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
