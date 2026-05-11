"""
setup_volume_beam.py — Upload training data to Beam Cloud Volume.

Run this ONCE before training. Uploads local preprocessed data files
to the persistent Beam Volume so all three training stages can access them.

The correct beam cp syntax is:
    beam cp <local_file_or_dir> beam://<volume-name>
Beam places the item at the root of the volume, preserving its local name.

Prerequisites
─────────────
    pip install beam-client
    beam configure default --token <YOUR_TOKEN>
    beam volume create gector-data          # only needed the very first time

Usage
─────
    python setup_volume_beam.py

Optional args:
    python setup_volume_beam.py --vocab-path data/output_vocabulary
    python setup_volume_beam.py --skip-stage 2   # skip uploading stage 2 data
    python setup_volume_beam.py --check-only      # just list what's already there

Volume layout after upload
──────────────────────────
beam://gector-data/
├── output_vocabulary/      ← vocab dir (uploaded as-is from local)
├── stage1.train
├── stage1.dev
├── stage2.train
├── stage2.dev
├── stage3.train
└── stage3.dev

Inside the container (mount_path="./gector-data"), these are accessible at:
    ./gector-data/output_vocabulary/
    ./gector-data/stage1.train
    … etc.

beam_train.py sets DATA = f"{MOUNT}/data" with MOUNT = "./gector-data".
The paths there are updated to match this flat layout (no extra data/ subdir
on the volume side — files live directly at the volume root).
"""

import argparse
import subprocess
import sys
from pathlib import Path

VOLUME_NAME = "gector-data"
VOLUME_URI  = f"beam://{VOLUME_NAME}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def beam_cp(local: Path, label: str) -> bool:
    """
    Upload one file or directory to the root of the Beam volume.

    Correct syntax:  beam cp <local>  beam://<volume>
    Beam names the destination after the local file/dir automatically.
    Do NOT append a subpath — that causes the 'unable to find volume' error.
    """
    cmd = ["beam", "cp", str(local), VOLUME_URI]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: upload failed for '{local}' (exit code {result.returncode})")
        return False
    return True


def size_str(path: Path) -> str:
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total = path.stat().st_size
    mb = total / 1e6
    return f"{mb:.1f} MB" if mb < 1000 else f"{mb / 1000:.2f} GB"


def check_volume():
    print(f"\nContents of {VOLUME_URI} :")
    subprocess.run(["beam", "ls", VOLUME_NAME])


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    if args.check_only:
        check_volume()
        return

    print(f"Uploading GECToR training data to Beam Volume '{VOLUME_NAME}' …\n")
    print("NOTE: files are uploaded to the volume root.")
    print(f"      Inside the container they will be at ./gector-data/<filename>\n")

    # (local_path_string, human_label)
    to_upload = [
        (args.vocab_path, "vocab dir (output_vocabulary)"),
    ]

    for stage in [1, 2, 3]:
        if args.skip_stage == stage:
            print(f"  Skipping stage {stage} data (--skip-stage {stage})")
            continue
        to_upload += [
            (getattr(args, f"stage{stage}_train"), f"stage{stage} train"),
            (getattr(args, f"stage{stage}_dev"),   f"stage{stage} dev"),
        ]

    skipped  = []
    failed   = []
    uploaded = []

    for local_str, label in to_upload:
        local = Path(local_str)
        if not local.exists():
            print(f"  SKIP (not found locally): {local_str}")
            skipped.append(local_str)
            continue

        print(f"\n  [{label}]  {local_str}  ({size_str(local)})")
        if beam_cp(local, label):
            uploaded.append(local_str)
        else:
            failed.append(local_str)

    # Summary
    print("\n" + "─" * 50)
    print(f"✓ Uploaded : {len(uploaded)} item(s)")
    if skipped:
        print(f"⚠ Skipped  : {len(skipped)} (files not found locally)")
        for s in skipped:
            print(f"    {s}")
    if failed:
        print(f"✗ Failed   : {len(failed)}")
        for f in failed:
            print(f"    {f}")
        sys.exit(1)

    print("\nVerifying volume contents …")
    check_volume()

    print("\nReady to train. Run stages in order:")
    print("  python beam_train.py preprocess")
    print("  python beam_train.py stage1")
    print("  python beam_train.py stage2")
    print("  python beam_train.py stage3")


# ── CLI ────────────────────────────────────────────────────────────────────────

def get_parser():
    parser = argparse.ArgumentParser(
        description="Upload GECToR training data to a Beam Cloud Volume"
    )
    parser.add_argument("--vocab-path",    default="data/output_vocabulary",
                        dest="vocab_path",
                        help="Local path to the output_vocabulary directory")
    parser.add_argument("--stage1-train",  default="data/stage1.train",  dest="stage1_train")
    parser.add_argument("--stage1-dev",    default="data/stage1.dev",    dest="stage1_dev")
    parser.add_argument("--stage2-train",  default="data/stage2.train",  dest="stage2_train")
    parser.add_argument("--stage2-dev",    default="data/stage2.dev",    dest="stage2_dev")
    parser.add_argument("--stage3-train",  default="data/stage3.train",  dest="stage3_train")
    parser.add_argument("--stage3-dev",    default="data/stage3.dev",    dest="stage3_dev")
    parser.add_argument("--skip-stage",    type=int, default=0,
                        dest="skip_stage",
                        help="Skip uploading data for this stage number (1, 2, or 3)")
    parser.add_argument("--check-only",    action="store_true",
                        dest="check_only",
                        help="Only list what's in the volume; don't upload")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)