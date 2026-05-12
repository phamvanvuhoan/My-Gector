"""
repair_shards.py — Verify, patch, and rebuild shard files on Beam.

Modes
─────
verify (default)
    - Tries to load every shard with torch.load.
    - Deletes any corrupted shard (so preprocess can regenerate it).
    - Adds the missing 'n' key to healthy shards.
    - Rebuilds manifests, excluding deleted shards.
    After this, re-run:  python beam_train.py preprocess
    Preprocessing will skip all healthy shards and only redo the deleted ones.

dry-run
    - Same as verify but prints what would be deleted/patched without writing.

Usage
─────
    python repair_shards.py            # verify + patch + rebuild manifests
    python repair_shards.py --dry-run  # print plan, write nothing
"""

import argparse
import sys
from beam import Image, Volume, function

VOLUME_NAME = "gector-data"
MOUNT       = "./gector-data"

gector_volume = Volume(name=VOLUME_NAME, mount_path=MOUNT)

repair_image = (
    Image(python_version="python3.11")
    .add_python_packages(["torch>=2.6.0"])
)


@function(
    image   = repair_image,
    cpu     = 2,
    memory  = "16Gi",
    volumes = [gector_volume],
    timeout = 3600,
)
def repair(data_dir: str = MOUNT, dry_run: bool = False):
    import re
    import gc
    import os
    from collections import defaultdict
    from pathlib import Path
    import torch

    # ── helpers ────────────────────────────────────────────────────────────────

    def find_shard_files(data_dir):
        pattern = re.compile(r'\.cache_([0-9a-f]{8})_shard(\d{4})\.pt$')
        groups  = defaultdict(list)
        for p in Path(data_dir).rglob("*.pt"):
            if '_manifest' in p.name:
                continue
            m = pattern.search(p.name)
            if m:
                cache_hash   = m.group(1)
                suffix_start = p.name.index(f'.cache_{cache_hash}_shard')
                source_stem  = p.name[:suffix_start]
                group_key    = (str(p.parent), source_stem, cache_hash)
                groups[group_key].append(p)
        for key in groups:
            groups[key].sort(
                key=lambda p: int(re.search(r'shard(\d{4})', p.name).group(1))
            )
        return groups

    def manifest_path_for(shard_paths):
        first = Path(shard_paths[0])
        name  = re.sub(r'_shard\d{4}\.pt$', '_manifest.pt', first.name)
        return first.parent / name

    def try_load_shard(shard_path):
        """
        Returns (data, error_str).
        Loads the full file — we need to verify the tensor payload too,
        not just the header. If corrupted, returns (None, error).
        """
        try:
            data = torch.load(shard_path, weights_only=False)
            # Basic sanity: srcs must exist and be non-empty
            if 'srcs' not in data or len(data['srcs']) == 0:
                return None, "missing or empty 'srcs'"
            return data, None
        except Exception as e:
            return None, str(e)

    # ── main logic ─────────────────────────────────────────────────────────────

    groups = find_shard_files(data_dir)

    if not groups:
        print(f"No shard files found under '{data_dir}'")
        return

    total_shards = sum(len(v) for v in groups.values())
    print(f"Found {total_shards} shard(s) across {len(groups)} group(s)")
    if dry_run:
        print("DRY RUN — nothing will be written or deleted.\n")

    for (parent, stem, cache_hash), shard_paths in sorted(groups.items()):
        print(f"\n── {stem}  [hash={cache_hash}]  ({len(shard_paths)} shards)")

        healthy_paths   = []
        shard_lengths   = []
        n_corrupted     = 0
        n_already_ok    = 0
        n_patched       = 0

        for i, p in enumerate(shard_paths):
            print(f"  [{i+1:02d}/{len(shard_paths)}] {p.name} … ", end="", flush=True)

            data, err = try_load_shard(p)

            if err:
                print(f"CORRUPTED ({err})")
                print(f"    → {'would delete' if dry_run else 'deleting'} {p.name}")
                if not dry_run:
                    os.remove(p)
                n_corrupted += 1
                # Don't add to healthy_paths — preprocess will regenerate this shard
                continue

            n = len(data['srcs'])

            if 'n' not in data:
                data['n'] = n
                if not dry_run:
                    torch.save(data, p)
                status = "patched 'n' ✓"
                n_patched += 1
            else:
                status = "ok"
                n_already_ok += 1

            healthy_paths.append(p)
            shard_lengths.append(n)
            print(f"n={n:,}  {status}")

            del data
            gc.collect()

        # Rebuild manifest with only healthy shards
        if healthy_paths:
            manifest_path = manifest_path_for(healthy_paths)
            manifest = {
                'shard_paths':   [str(p) for p in healthy_paths],
                'shard_lengths': shard_lengths,
            }
            if not dry_run:
                torch.save(manifest, manifest_path)
            print(f"\n  Manifest → {manifest_path.name}")
            print(f"  Healthy shards : {len(healthy_paths)}")
            print(f"  Corrupted/deleted: {n_corrupted}")
            print(f"  Total sentences  : {sum(shard_lengths):,}")
        else:
            print(f"\n  All shards corrupted — manifest not written.")
            print(f"  Re-run preprocess from scratch for this file.")

        if n_corrupted > 0 and not dry_run:
            print(f"\n  ⚠  {n_corrupted} shard(s) deleted.")
            print(f"     Re-run:  python beam_train.py preprocess")
            print(f"     Only the missing shard(s) will be recomputed.")

    print("\n✓ Done.")


# ── local entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without writing or deleting anything")
    args = parser.parse_args()

    repair.remote(data_dir=MOUNT, dry_run=args.dry_run)