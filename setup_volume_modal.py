"""
modal_setup_volume.py — Upload training data to Modal Volume.

Run this ONCE before training. Uploads local preprocessed data files
to the persistent Modal Volume so all three training stages can access them.

Usage:
    modal run modal_setup_volume.py

Optional args:
    modal run modal_setup_volume.py --vocab-path data/output_vocabulary
    modal run modal_setup_volume.py --skip-stage 2  # skip uploading stage 2 data
"""

import os
from pathlib import Path
import modal

app    = modal.App("gector-setup")
volume = modal.Volume.from_name("gector-data", create_if_missing=True)
MOUNT  = "/gector-data"

# ── Upload helper (runs on Modal so large files go cloud-to-cloud fast) ───────

image = modal.Image.debian_slim().pip_install("tqdm")

@app.function(
    image=image,
    volumes={MOUNT: volume},
    timeout=3600,
)
def check_volume():
    """List what's already in the volume."""
    import subprocess
    result = subprocess.run(["find", MOUNT, "-type", "f"], capture_output=True, text=True)
    files = result.stdout.strip().split("\n") if result.stdout.strip() else []
    print(f"Volume contains {len(files)} files:")
    for f in sorted(files)[:50]:
        size = Path(f).stat().st_size / 1e6
        print(f"  {f}  ({size:.1f} MB)")
    if len(files) > 50:
        print(f"  ... and {len(files)-50} more")

# ── Local entrypoint — does the actual upload from your machine ───────────────

@app.local_entrypoint()
def main(
    vocab_path:    str = "data/output_vocabulary",
    stage1_train:  str = "data/stage1.train",
    stage1_dev:    str = "data/stage1.dev",
    stage2_train:  str = "data/stage2.train",
    stage2_dev:    str = "data/stage2.dev",
    stage3_train:  str = "data/stage3.train",
    stage3_dev:    str = "data/stage3.dev",
    skip_stage:    int = 0,   # set to 1/2/3 to skip uploading that stage's data
):
    """
    Uploads all training data to the Modal Volume.

    Modal volumes support direct writes via the Python SDK — no CLI needed.
    Large files (stage1 is ~3-5 GB) will take a few minutes.
    """
    print("Uploading GECToR training data to Modal Volume 'gector-data'...\n")

    files_to_upload = [
        # (local_path, remote_path_in_volume)
        (vocab_path, "data/output_vocabulary"),
    ]

    for stage in [1, 2, 3]:
        if skip_stage == stage:
            print(f"Skipping stage {stage} data (--skip-stage {stage})")
            continue
        train = locals().get(f"stage{stage}_train")
        dev   = locals().get(f"stage{stage}_dev")
        if train:
            files_to_upload.append((train, f"data/stage{stage}.train"))
        if dev:
            files_to_upload.append((dev,   f"data/stage{stage}.dev"))

    with volume.batch_upload() as batch:
        for local_path, remote_path in files_to_upload:
            local = Path(local_path)
            if not local.exists():
                print(f"  SKIP (not found locally): {local_path}")
                continue
 
            if local.is_dir():
                size_mb = sum(
                    f.stat().st_size for f in local.rglob("*") if f.is_file()
                ) / 1e6
                print(f"  Uploading dir  {local_path}/ ({size_mb:.1f} MB) → {remote_path}/")
                batch.put_directory(str(local), remote_path)
            else:
                size_mb = local.stat().st_size / 1e6
                print(f"  Uploading file {local_path} ({size_mb:.1f} MB) → {remote_path}")
                batch.put_file(str(local), remote_path)

    print("\n✓ Upload complete.")
    print("\nVerifying volume contents...")
    check_volume.remote()

    print("\nReady to train. Run stages in order:")
    print("  modal run modal_train.py::run_stage1")
    print("  modal run modal_train.py::run_stage2")
    print("  modal run modal_train.py::run_stage3")