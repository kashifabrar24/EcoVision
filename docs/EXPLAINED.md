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

