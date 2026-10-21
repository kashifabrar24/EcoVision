"""Robust download utility for TrashNet from Hugging Face.

Streams dataset archives directly from Hugging Face with auto-resume,
progress reporting, and cache linking so load_dataset('garythung/trashnet')
can load all 5,054 rows instantly without stalls.
"""

import os
import sys
import time
from pathlib import Path
import requests

HF_REPO = "garythung/trashnet"
REVISION = "94cd17ebfd6702bf62281d3c89f22ff649815b46"

ORIGINAL_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{REVISION}/dataset-original.zip"
RESIZED_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{REVISION}/dataset-resized.zip"

ORIGINAL_SIZE = 3634187193
ORIGINAL_BLOB_HASH = "4e46374d6da507b3ac5934a0e7769c21235d9ed0e2329da72ca847f866e5f706"


def get_cache_paths():
    home = Path.home()
    hub_dir = home / ".cache" / "huggingface" / "hub" / f"datasets--{HF_REPO.replace('/', '--')}"
    blobs_dir = hub_dir / "blobs"
    snapshots_dir = hub_dir / "snapshots" / REVISION
    blobs_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    return blobs_dir, snapshots_dir


def download_with_resume(url: str, target_path: Path, expected_size: int) -> bool:
    target_part = target_path.with_suffix(target_path.suffix + ".part")
    existing_bytes = target_part.stat().st_size if target_part.exists() else 0

    if target_path.exists() and target_path.stat().st_size == expected_size:
        print(f"[CACHE HIT] {target_path.name} already complete ({expected_size / (1024**2):.1f} MB)")
        return True

    headers = {}
    mode = "wb"
    if existing_bytes > 0:
        if existing_bytes < expected_size:
            headers["Range"] = f"bytes={existing_bytes}-"
            mode = "ab"
            print(f"[RESUME] Resuming download from {existing_bytes / (1024**2):.1f} MB...")
        else:
            target_part.replace(target_path)
            return True

    print(f"Connecting to: {url}")
    session = requests.Session()
    session.headers.update({"User-Agent": "EcoVision-Data-Audit/1.0"})

    with session.get(url, headers=headers, stream=True, timeout=30) as resp:
        if resp.status_code not in (200, 206):
            print(f"[ERROR] HTTP {resp.status_code} while connecting to {url}")
            return False

        chunk_size = 4 * 1024 * 1024  # 4MB chunks
        downloaded = existing_bytes
        last_log_time = time.time()
        start_time = time.time()
        bytes_since_log = 0

        with open(target_part, mode) as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                bytes_since_log += len(chunk)

                now = time.time()
                if now - last_log_time >= 8.0:
                    speed_mb = (bytes_since_log / (1024**2)) / (now - last_log_time)
                    pct = (downloaded / expected_size) * 100
                    eta_sec = (expected_size - downloaded) / (speed_mb * 1024 * 1024) if speed_mb > 0 else 0
                    print(
                        f"Progress: {downloaded / (1024**2):.1f} / {expected_size / (1024**2):.1f} MB "
                        f"({pct:.1f}%) | Speed: {speed_mb:.2f} MB/s | ETA: {eta_sec / 60:.1f} min",
                        flush=True,
                    )
                    last_log_time = now
                    bytes_since_log = 0

    if target_part.stat().st_size == expected_size:
        if target_path.exists():
            target_path.unlink()
        target_part.replace(target_path)
        print(f"[COMPLETE] Finished downloading {target_path.name}")
        return True
    else:
        print(f"[WARNING] Size mismatch: got {target_part.stat().st_size}, expected {expected_size}")
        return False


def link_blob_to_snapshot(blob_path: Path, snapshot_path: Path):
    if snapshot_path.exists() or snapshot_path.is_symlink():
        try:
            snapshot_path.unlink()
        except Exception:
            pass

    # On Windows, try hardlink first; fallback to copy if privileges deny
    try:
        os.link(blob_path, snapshot_path)
        print(f"[LINKED] Hardlinked blob to snapshot: {snapshot_path.name}")
    except Exception as e:
        print(f"[FALLBACK] Copying blob to snapshot ({e})...")
        import shutil
        shutil.copyfile(blob_path, snapshot_path)
        print(f"[COPIED] Blob to snapshot: {snapshot_path.name}")


def main():
    print("=" * 60)
    print("EcoVision - Downloading TrashNet Dataset Archives")
    print("=" * 60)
    blobs_dir, snapshots_dir = get_cache_paths()
    blob_target = blobs_dir / ORIGINAL_BLOB_HASH
    snapshot_target = snapshots_dir / "dataset-original.zip"

    success = download_with_resume(ORIGINAL_URL, blob_target, ORIGINAL_SIZE)
    if not success:
        print("[ERROR] Failed to download dataset-original.zip")
        return 1

    link_blob_to_snapshot(blob_target, snapshot_target)

    # Verify that dataset-resized.zip snapshot exists as well
    snapshot_resized = snapshots_dir / "dataset-resized.zip"
    if not snapshot_resized.exists():
        print("Downloading dataset-resized.zip...")
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo_id=HF_REPO, filename="dataset-resized.zip", repo_type="dataset")

    print("=" * 60)
    print("[SUCCESS] All TrashNet archives successfully staged in Hugging Face cache.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
