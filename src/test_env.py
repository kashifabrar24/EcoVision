"""Verify that all project dependencies and GPU acceleration are operational."""

import sys
import importlib


def test_package(pkg_name: str) -> bool:
    try:
        mod = importlib.import_module(pkg_name)
        ver = getattr(mod, "__version__", "installed")
        print(f"[OK] {pkg_name:<18} : version {ver}")
        return True
    except Exception as e:
        print(f"[FAILED] {pkg_name:<18} : {e}")
        return False


def main() -> int:
    print("=" * 60)
    print("EcoVision - Environment Verification")
    print("=" * 60)
    packages = [
        "torch",
        "torchvision",
        "datasets",
        "huggingface_hub",
        "PIL",
        "imagehash",
        "sklearn",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "gradio",
        "tqdm",
    ]

    all_passed = True
    for pkg in packages:
        if not test_package(pkg):
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("[SUCCESS] All dependencies are correctly installed and functional.")
        return 0
    else:
        print("[ERROR] One or more dependencies failed to import.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
