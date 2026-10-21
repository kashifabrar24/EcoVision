"""Check CUDA availability and report GPU hardware specs for EcoVision.

Ensures training runs on NVIDIA GPU (RTX 3050) and never silently falls back to CPU.
"""

import sys
import torch


def main() -> int:
    print("=" * 60)
    print("EcoVision - Hardware Environment & CUDA Check")
    print("=" * 60)
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n[CRITICAL ERROR] CUDA is NOT available to PyTorch!")
        print("EcoVision requires GPU acceleration (RTX 3050). Training on CPU is forbidden.")
        return 1

    gpu_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    total_memory_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024**3)

    print(f"PyTorch Built with CUDA: {torch.version.cuda}")
    print(f"Active GPU Device ID    : {current_device} (Total GPUs: {gpu_count})")
    print(f"GPU Model Name          : {device_name}")
    print(f"Total Dedicated VRAM    : {total_memory_gb:.2f} GB")
    print("=" * 60)
    print("[SUCCESS] GPU verification passed. Ready for training.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
