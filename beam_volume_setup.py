"""
setup_volume_beam.py — Upload training data to Beam Cloud Volume.

Run this ONCE before training. Uploads local preprocessed data files
to the persistent Beam Volume so all three training stages can access them.

Unlike Modal (which has a Python batch_upload API), Beam's volume uploads
go through the CLI `beam cp` command. This script wraps those calls
with progress reporting and error checking so the experience is equivalent.

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
"""

import argparse
import subprocess
import sys
from pathlib import Path

VOLUME_NAME = "gector-data"


# ── Helpers ────────────────────────────────────────────────────────────────────

def beam_cp(local: Path, remote: str, label: str) -> bool:
    """
    Upload one file or directory to the Beam volume.
    Returns True on success, False on failure.
    """
    dest = f"beam://{VOLUME_NAME}/{remote}"
    print(f"  Uploading {label} → {dest}")
    result = subprocess.run(
        ["beam", "cp", str(local), dest],
        capture_output=False,   # stream output so large files show progress
    )
    if result.returncode != 0:
        print(f"  ERROR: upload failed for {local} (exit code {result.returncode})")
        return False
    return True


def size_str(path: Path) -> str:
    """Human-readable size for a file or directory."""
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    else:
        total = path.stat().st_size
    mb = total / 1e6
    return f"{mb:.1f} MB" if mb < 1000 else f"{mb/1000:.2f} GB"


def check_volume():
    """List all files currently in the volume (uses beam ls)."""
    print(f"\nContents of beam://{VOLUME_NAME} :")
    result = subprocess.run(
        ["beam", "ls", VOLUME_NAME],
        capture_output=False,
    )
    if result.returncode != 0:
        print("  (Could not list volume — is the volume created yet?)")


# ── Main upload logic ──────────────────────────────────────────────────────────

def main(args):
    if args.check_only:
        check_volume()
        return

    print(f"Uploading GECToR training data to Beam Volume '{VOLUME_NAME}' …\n")

    # Build the list of (local_path, remote_relative_path, label) tuples
    to_upload = [
        (args.vocab_path, "data/output_vocabulary", "vocab dir"),
    ]

    for stage in [1, 2, 3]:
        if args.skip_stage == stage:
            print(f"  Skipping stage {stage} data (--skip-stage {stage})")
            continue
        to_upload += [
            (getattr(args, f"stage{stage}_train"), f"data/stage{stage}.train", f"stage{stage} train"),
            (getattr(args, f"stage{stage}_dev"),   f"data/stage{stage}.dev",   f"stage{stage} dev"),
        ]

    # Upload each item
    skipped  = []
    failed   = []
    uploaded = []

    for local_str, remote, label in to_upload:
        local = Path(local_str)
        if not local.exists():
            print(f"  SKIP (not found locally): {local_str}")
            skipped.append(local_str)
            continue

        print(f"\n  [{label}]  {local_str}  ({size_str(local)})")
        ok = beam_cp(local, remote, label)
        if ok:
            uploaded.append(local_str)
        else:
            failed.append(local_str)

    # Summary
    print("\n" + "─" * 50)
    print(f"✓ Uploaded : {len(uploaded)} item(s)")
    if skipped:
        print(f"⚠ Skipped  : {len(skipped)} item(s) (files not found locally)")
        for s in skipped:
            print(f"    {s}")
    if failed:
        print(f"✗ Failed   : {len(failed)} item(s)")
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
                        help="Local path to the output_vocabulary directory")
    parser.add_argument("--stage1-train",  default="data/stage1.train")
    parser.add_argument("--stage1-dev",    default="data/stage1.dev")
    parser.add_argument("--stage2-train",  default="data/stage2.train")
    parser.add_argument("--stage2-dev",    default="data/stage2.dev")
    parser.add_argument("--stage3-train",  default="data/stage3.train")
    parser.add_argument("--stage3-dev",    default="data/stage3.dev")
    parser.add_argument("--skip-stage",    type=int, default=0,
                        help="Skip uploading data for this stage number (1, 2, or 3)")
    parser.add_argument("--check-only",    action="store_true",
                        help="Only list what's already in the volume; don't upload anything")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args   = parser.parse_args()
    main(args)