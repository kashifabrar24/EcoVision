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
1. **Q: Why is the 4.33:1 class imbalance between paper (23.51%) and trash (5.42%) problematic for evaluation?**
   - **A:** If evaluated using overall accuracy alone, an algorithm could predict the majority class and achieve ~75-80% accuracy while failing 100% of the time on miscellaneous trash. Macro-averaged metrics (macro-F1) weight all classes equally, exposing poor performance on minority classes.
2. **Q: How does the discovery of cross-class duplicates (e.g. `glass115.jpg` and `metal91.jpg`) affect model training?**
   - **A:** Cross-class duplicates represent conflicting label noise. If identical images are assigned contradictory labels, the loss function penalizes correct predictions and confuses gradients. In addition, if duplicates land in different splits, test set leakage occurs.
3. **Q: Why was `dataset-original.zip` paired with `dataset-resized.zip` on Hugging Face?**
   - **A:** The original TrashNet repository contained both raw high-resolution captures (up to 4000px) and a convenience downscaled set (`resize.py` output to 512×384). Hugging Face mirrored both archives without realizing its automatic dataset loader would combine both into the same split.




