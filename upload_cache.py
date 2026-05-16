"""
upload_cache_to_volume.py — Upload local cache files to a Modal Volume.

Your friend runs this ONCE in their own Modal account to populate their
volume with your preprocessed cache, skipping preprocess_all entirely.

Usage:
    modal run upload_cache_to_volume.py --local-dir ./gector_cache
    modal run upload_cache_to_volume.py --local-dir ./gector_cache --stage 1
    modal run upload_cache_to_volume.py --local-dir ./gector_cache --volume-name gector-data
"""

import os
from pathlib import Path
import modal

app = modal.App("gector-upload-cache")

VALID_SUFFIXES = {
    ".input_ids.mmap",
    ".attention_mask.mmap",
    ".word_masks.mmap",
    ".labels.mmap",
    ".d_labels.mmap",
    ".meta.pt",
}

image = modal.Image.debian_slim().pip_install("tqdm")


@app.function(
    image   = image,
    volumes = {},        # volume attached dynamically in main()
    timeout = 3600,
)
def verify_volume(volume_name: str, cache_dir: str):
    """List uploaded files to verify the upload succeeded."""
    import subprocess
    mount   = "/gector-data"
    volume  = modal.Volume.from_name(volume_name, create_if_missing=False)
    result  = subprocess.run(
        ["find", f"{mount}/{cache_dir}", "-type", "f"],
        capture_output=True, text=True
    )
    files = result.stdout.strip().split("\n") if result.stdout.strip() else []
    total_gb = sum(
        Path(f).stat().st_size for f in files if Path(f).exists()
    ) / (1024 ** 3)
    print(f"\nVolume '{volume_name}' cache contains {len(files)} files ({total_gb:.2f} GB)")
    for f in sorted(files):
        size_mb = Path(f).stat().st_size / 1e6 if Path(f).exists() else 0
        print(f"  {Path(f).name}  ({size_mb:.1f} MB)")


@app.local_entrypoint()
def main(
    local_dir:   str = "./gector_cache",  # local folder from download_cache_from_volume.py
    cache_dir:   str = "cache",           # subfolder inside the volume
    volume_name: str = "gector-data",     # Modal volume name (create if missing)
    stage:       int = 0,                 # 0 = all stages, 1/2/3 = specific stage
    verify:      bool = True,             # verify after upload
):
    """
    Upload local cache files to a Modal Volume.
    Safe to re-run — already-uploaded files are skipped by batch_upload.
    """
    local_root = Path(local_dir)
    if not local_root.exists():
        print(f"ERROR: local_dir not found: {local_dir}")
        print("Run download_cache_from_volume.py first.")
        return

    # Attach (or create) the volume in caller's Modal account
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)

    # Collect files to upload
    all_files = sorted(local_root.iterdir())
    files_to_upload = []
    skipped = []

    for f in all_files:
        if not f.is_file():
            continue

        # Skip progress files
        if f.name.endswith(".progress.pkl"):
            skipped.append(f.name)
            continue

        # Skip non-cache files
        if not any(f.name.endswith(s) for s in VALID_SUFFIXES):
            skipped.append(f.name)
            continue

        # Filter by stage if requested
        if stage > 0 and f"stage{stage}" not in f.name:
            continue

        files_to_upload.append(f)

    if not files_to_upload:
        print(f"No valid cache files found in {local_dir}")
        if stage > 0:
            print(f"(filtered to stage {stage} only)")
        return

    # Print summary
    total_bytes = sum(f.stat().st_size for f in files_to_upload)
    total_gb    = total_bytes / (1024 ** 3)

    print(f"Uploading to Modal Volume '{volume_name}' ...")
    print(f"  Files : {len(files_to_upload)}")
    print(f"  Size  : {total_gb:.2f} GB")
    if stage > 0:
        print(f"  Stage : {stage} only")
    if skipped:
        print(f"  Skipped (not cache files): {len(skipped)}")
    print()

    # Group by stage for readable progress
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for f in files_to_upload:
        parts = f.name.split(".cache_")
        key   = parts[0] if len(parts) == 2 else "other"
        groups[key].append(f)

    remote_cache = f"{cache_dir}"   # relative to volume mount

    with volume.batch_upload(force=False) as batch:
        for group_key, group_files in sorted(groups.items()):
            group_gb = sum(f.stat().st_size for f in group_files) / (1024 ** 3)
            print(f"  {group_key:<40} {len(group_files):>2} files  {group_gb:.2f} GB")
            for f in group_files:
                remote_path = f"{remote_cache}/{f.name}"
                batch.put_file(str(f), remote_path)

    print("\n✓ Upload complete.")
    print(f"\nYour friend's volume '{volume_name}' is ready.")
    print("\nThey can now run training directly:")
    print("  modal run modal_train.py::run_stage1")
    print("  modal run modal_train.py::run_stage2")
    print("  modal run modal_train.py::run_stage3")