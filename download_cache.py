"""
download_cache_from_volume.py — Download preprocessed cache from Modal Volume to local disk.

Run this to get the cache files from your Modal Volume onto your machine,
so you can then upload them to a friend's volume or to HuggingFace.

Usage:
    modal run download_cache_from_volume.py
    modal run download_cache_from_volume.py --local-dir ./my_cache
    modal run download_cache_from_volume.py --stage 1  # only download stage 1 cache
"""

import os
from pathlib import Path
import modal

app    = modal.App("gector-download-cache")
volume = modal.Volume.from_name("gector-data", create_if_missing=False)
MOUNT  = "/"

VALID_SUFFIXES = {
    ".input_ids.mmap",
    ".attention_mask.mmap",
    ".word_masks.mmap",
    ".labels.mmap",
    ".d_labels.mmap",
    ".meta.pt",
}


@app.local_entrypoint()
def main(
    local_dir:  str = "./gector_cache",   # where to save on your machine
    cache_dir:  str = "cache",            # subfolder inside the volume
    stage:      int = 0,                  # 0 = all stages, 1/2/3 = specific stage
):
    """
    Download cache files from Modal Volume to local disk.
    Progress files (.progress.pkl) are excluded — not needed downstream.
    """
    local_root = Path(local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    remote_cache = os.path.join(MOUNT, cache_dir)

    print(f"Scanning volume at {remote_cache} ...")

    # List all files in the cache directory
    try:
        all_entries = list(volume.listdir(remote_cache))
    except Exception as e:
        print(f"ERROR: Could not list {remote_cache}: {e}")
        print("Have you run preprocess_all yet?")
        return

    # Filter to valid cache files only
    entries = []
    for entry in all_entries:
        name = Path(entry.path).name

        # Skip progress files
        if name.endswith(".progress.pkl"):
            continue

        # Skip if not a valid cache suffix
        if not any(name.endswith(s) for s in VALID_SUFFIXES):
            continue

        # Filter by stage if requested
        if stage > 0 and f"stage{stage}" not in name:
            continue

        entries.append(entry)

    if not entries:
        print("No cache files found. Check --stage or run preprocess_all first.")
        return

    # Print summary
    total_bytes = sum(e.size for e in entries if hasattr(e, "size"))
    total_gb    = total_bytes / (1024 ** 3)
    print(f"\nFound {len(entries)} files ({total_gb:.2f} GB)")
    if stage > 0:
        print(f"Filtered to stage {stage} only")
    print()

    # Download each file
    for i, entry in enumerate(entries):
        name      = Path(entry.path).name
        dest_path = local_root / name
        size_mb   = getattr(entry, "size", 0) / 1e6

        if dest_path.exists():
            local_size = dest_path.stat().st_size
            remote_size = getattr(entry, "size", 0)
            if local_size == remote_size:
                print(f"  SKIP (already exists): {name}")
                continue

        print(f"  [{i+1}/{len(entries)}] {name} ({size_mb:.1f} MB) ...", end=" ", flush=True)

        with dest_path.open("wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)

        print("✓")

    print(f"\n✓ Download complete → {local_root.resolve()}")
    print(f"\nNext step — upload to a new volume:")
    print(f"  modal run upload_cache_to_volume.py --local-dir {local_dir}")