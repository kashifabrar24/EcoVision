"""Phase 8: Inference Latency Benchmark for EcoVision MobileNetV2.

Measures median and p95 latency (in milliseconds) for single-image inference:
- CPU and GPU devices
- Model-only and End-to-End (image load + preprocess + model inference)
- 10 warmup iterations, 100 timed iterations
- torch.cuda.synchronize() on GPU
- Model file size and parameter count
Outputs results to reports/latency.json.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from src.models import create_mobilenetv2_transfer
from src.predict import preprocess_image


def benchmark_device(
    model: nn.Module,
    sample_path: Path,
    device: torch.device,
    n_warmup: int = 10,
    n_timed: int = 100,
) -> Dict:
    """Benchmark model-only and end-to-end inference latency on a given device."""
    model = model.to(device)
    model.eval()

    is_cuda = (device.type == "cuda")

    # Pre-load sample image for model-only benchmark
    sample_img = Image.open(sample_path)
    tensor_sample = preprocess_image(sample_img).to(device)

    # -------------------------------------------------------------
    # 1. Model-Only Latency
    # -------------------------------------------------------------
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            if is_cuda:
                torch.cuda.synchronize()
            _ = model(tensor_sample)
            if is_cuda:
                torch.cuda.synchronize()

    # Timed runs
    model_times_ms = []
    with torch.no_grad():
        for _ in range(n_timed):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(tensor_sample)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            model_times_ms.append((t1 - t0) * 1000.0)

    # -------------------------------------------------------------
    # 2. End-to-End Latency (Image.open + preprocess + inference)
    # -------------------------------------------------------------
    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            if is_cuda:
                torch.cuda.synchronize()
            img = Image.open(sample_path)
            x = preprocess_image(img).to(device)
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()

    # Timed runs
    e2e_times_ms = []
    with torch.no_grad():
        for _ in range(n_timed):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            img = Image.open(sample_path)
            x = preprocess_image(img).to(device)
            _ = model(x)
            if is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            e2e_times_ms.append((t1 - t0) * 1000.0)

    return {
        "model_only_ms": {
            "median": round(float(np.median(model_times_ms)), 2),
            "p95": round(float(np.percentile(model_times_ms, 95)), 2),
            "mean": round(float(np.mean(model_times_ms)), 2),
            "std": round(float(np.std(model_times_ms)), 2),
            "min": round(float(np.min(model_times_ms)), 2),
            "max": round(float(np.max(model_times_ms)), 2),
        },
        "end_to_end_ms": {
            "median": round(float(np.median(e2e_times_ms)), 2),
            "p95": round(float(np.percentile(e2e_times_ms, 95)), 2),
            "mean": round(float(np.mean(e2e_times_ms)), 2),
            "std": round(float(np.std(e2e_times_ms)), 2),
            "min": round(float(np.min(e2e_times_ms)), 2),
            "max": round(float(np.max(e2e_times_ms)), 2),
        },
    }


def main():
    print("=" * 65)
    print("EcoVision - Phase 8: Inference Latency Benchmark")
    print("=" * 65)

    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = models_dir / "mobilenetv2_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {ckpt_path}")

    # Pick a representative test image
    test_image_dir = project_root / "data" / "processed" / "test"
    sample_images = list(test_image_dir.glob("*/*.jpg"))
    if not sample_images:
        raise FileNotFoundError(f"No test images found in {test_image_dir}")
    sample_path = sample_images[0]
    print(f"Sample test image: {sample_path.name}")

    # Model metadata
    file_size_bytes = os.path.getsize(ckpt_path)
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)

    model = create_mobilenetv2_transfer(num_classes=6, dropout_p=0.3, pretrained=False)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model file size : {file_size_mb} MB ({file_size_bytes:,} bytes)")
    print(f"Total params    : {total_params:,}")

    n_warmup = 10
    n_timed = 100

    results_payload = {
        "model_name": "MobileNetV2 Final",
        "checkpoint_path": str(ckpt_path.relative_to(project_root)),
        "file_size_mb": file_size_mb,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "benchmark_config": {
            "num_warmup_runs": n_warmup,
            "num_timed_runs": n_timed,
            "sample_image": sample_path.name,
        },
        "devices": {},
    }

    # 1. GPU Benchmark (if available)
    if torch.cuda.is_available():
        gpu_device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nBenchmarking on GPU ({gpu_name})...")
        gpu_metrics = benchmark_device(model, sample_path, gpu_device, n_warmup, n_timed)
        results_payload["devices"]["gpu"] = {
            "device_name": gpu_name,
            **gpu_metrics,
        }
        print(f"  GPU Model-Only : Median={gpu_metrics['model_only_ms']['median']} ms | p95={gpu_metrics['model_only_ms']['p95']} ms")
        print(f"  GPU End-to-End : Median={gpu_metrics['end_to_end_ms']['median']} ms | p95={gpu_metrics['end_to_end_ms']['p95']} ms")
    else:
        print("\nGPU not available, skipping GPU benchmark.")

    # 2. CPU Benchmark
    cpu_device = torch.device("cpu")
    cpu_name = platform.processor() or "CPU"
    print(f"\nBenchmarking on CPU ({cpu_name})...")
    cpu_metrics = benchmark_device(model, sample_path, cpu_device, n_warmup, n_timed)
    results_payload["devices"]["cpu"] = {
        "device_name": cpu_name,
        **cpu_metrics,
    }
    print(f"  CPU Model-Only : Median={cpu_metrics['model_only_ms']['median']} ms | p95={cpu_metrics['model_only_ms']['p95']} ms")
    print(f"  CPU End-to-End : Median={cpu_metrics['end_to_end_ms']['median']} ms | p95={cpu_metrics['end_to_end_ms']['p95']} ms")

    # Save to reports/latency.json
    out_json_path = reports_dir / "latency.json"
    with open(out_json_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[SAVED] Latency report saved to {out_json_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
