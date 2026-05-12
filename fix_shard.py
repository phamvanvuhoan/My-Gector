"""
repair_shards.py — Patch existing shard files and rebuild manifests on Beam.

Usage:
    python repair_shards.py            # patch and write
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
    memory  = "16Gi",   # one shard at a time, ~3 Gi each — 16 Gi is safe headroom
    volumes = [gector_volume],
    timeout = 3600,
)
def repair(data_dir: str = MOUNT, dry_run: bool = False):
    import re
    import gc
    import sys
    from collections import defaultdict
    from pathlib import Path
    import torch

    def find_shard_files(data_dir):
        pattern = re.compile(r'\.cache_([0-9a-f]{8})_shard(\d{4})\.pt$')
        groups  = defaultdict(list)
        for p in Path(data_dir).rglob("*.pt"):
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

    def patch_shard(shard_path, dry_run):
        data = torch.load(shard_path, weights_only=False)
        n    = len(data['srcs'])
        if 'n' not in data:
            data['n'] = n
            if not dry_run:
                torch.save(data, shard_path)
            status = "patched ✓"
        else:
            status = "already ok"
        del data
        gc.collect()
        return n, status

    # ── Run ────────────────────────────────────────────────────────────────────
    groups = find_shard_files(data_dir)

    if not groups:
        print(f"No shard files found under '{data_dir}'")
        return

    print(f"Found {sum(len(v) for v in groups.values())} shard(s) "
          f"across {len(groups)} group(s)")
    if dry_run:
        print("DRY RUN — nothing will be written.\n")

    for (parent, stem, cache_hash), shard_paths in sorted(groups.items()):
        print(f"\n── {stem}  [hash={cache_hash}]  ({len(shard_paths)} shards)")
        shard_lengths = []
        for i, p in enumerate(shard_paths):
            print(f"  [{i+1:02d}/{len(shard_paths)}] {p.name} … ", end="", flush=True)
            n, status = patch_shard(p, dry_run)
            shard_lengths.append(n)
            print(f"n={n:,}  {status}")

        manifest_path = manifest_path_for(shard_paths)
        manifest = {
            'shard_paths':   [str(p) for p in shard_paths],
            'shard_lengths': shard_lengths,
        }
        if not dry_run:
            torch.save(manifest, manifest_path)
        print(f"  Manifest → {manifest_path.name}")
        print(f"  Total sentences: {sum(shard_lengths):,}")

    print("\n✓ Done. Run training normally — no reprocessing needed.")


# ── Local entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repair.remote(data_dir=MOUNT, dry_run=args.dry_run)