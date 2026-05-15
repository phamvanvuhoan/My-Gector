"""
upload_cache_to_hub.py — Upload GECToR preprocessed cache to HuggingFace.

Usage (local, if volume is mounted):
    python upload_cache_to_hub.py --repo_id yourname/gector-cache --cache_dir /gector-data/cache

Usage (on Modal, recommended):
    modal run upload_cache_to_hub.py --repo_id yourname/gector-cache
"""

import argparse
import os
from pathlib import Path

import modal

# ── Modal setup (only used when running on Modal) ─────────────────────────────

app    = modal.App("gector-upload-cache")
volume = modal.Volume.from_name("gector-data", create_if_missing=False)
MOUNT  = "/gector-data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("huggingface-hub>=0.28.1")
)

# ── Core upload logic ─────────────────────────────────────────────────────────

def _collect_cache_files(cache_dir: str) -> list[Path]:
    """
    Collect all valid cache files under cache_dir.

    Valid extensions (from dataset.py):
        .input_ids.mmap
        .attention_mask.mmap
        .word_masks.mmap
        .labels.mmap
        .d_labels.mmap
        .meta.pt

    Progress files (.progress.pkl) are excluded — they are
    resume artifacts, not needed by the downloader.
    """
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    valid_suffixes = {
        ".input_ids.mmap",
        ".attention_mask.mmap",
        ".word_masks.mmap",
        ".labels.mmap",
        ".d_labels.mmap",
        ".meta.pt",
    }

    files = []
    for f in sorted(cache_path.iterdir()):
        if not f.is_file():
            continue
        # Match any valid suffix
        matched = any(f.name.endswith(suffix) for suffix in valid_suffixes)
        if matched:
            files.append(f)

    return files


def _print_summary(files: list[Path]) -> None:
    total_bytes = sum(f.stat().st_size for f in files)
    total_gb    = total_bytes / (1024 ** 3)

    # Group by stage prefix for readability
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for f in files:
        # e.g. "stage1.train.cache_abc1234" → group key "stage1.train"
        parts = f.name.split(".cache_")
        key   = parts[0] if len(parts) == 2 else "other"
        groups[key].append(f)

    print(f"\n{'='*60}")
    print(f"Cache files to upload: {len(files)} files, {total_gb:.2f} GB total")
    print(f"{'='*60}")
    for key, group_files in sorted(groups.items()):
        group_bytes = sum(f.stat().st_size for f in group_files)
        group_gb    = group_bytes / (1024 ** 3)
        print(f"  {key:<40} {len(group_files):>2} files  {group_gb:.2f} GB")
    print(f"{'='*60}\n")


def upload_cache(
    repo_id:   str,
    cache_dir: str,
    private:   bool = True,
    token:     str  = None,
) -> None:
    from huggingface_hub import HfApi, CommitOperationAdd

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HuggingFace token not found. "
            "Set HF_TOKEN env var or pass --token."
        )

    api = HfApi(token=token)

    # ── Create repo if it doesn't exist ──────────────────────────────
    print(f"Creating/verifying dataset repo: {repo_id} (private={private})")
    api.create_repo(
        repo_id   = repo_id,
        repo_type = "dataset",
        private   = private,
        exist_ok  = True,
    )

    # ── Collect files ─────────────────────────────────────────────────
    files = _collect_cache_files(cache_dir)
    if not files:
        raise RuntimeError(
            f"No cache files found in {cache_dir}. "
            "Have you run preprocess_all yet?"
        )
    _print_summary(files)

    # ── Check which files are already on the hub ──────────────────────
    print("Checking existing files on hub...")
    try:
        existing = {
            f.rfilename
            for f in api.list_repo_tree(repo_id, repo_type="dataset")
            if hasattr(f, "rfilename")
        }
    except Exception:
        existing = set()

    to_upload = [f for f in files if f.name not in existing]
    skipped   = len(files) - len(to_upload)

    if skipped > 0:
        print(f"Skipping {skipped} files already on hub.")
    if not to_upload:
        print("✓ All files already uploaded. Nothing to do.")
        return

    print(f"Uploading {len(to_upload)} files...\n")

    # ── Upload in batches to avoid hitting HF API limits ─────────────
    # Large mmap files (>5GB) must be uploaded individually via LFS.
    # HF hub handles LFS automatically; we just batch small files.
    BATCH_SIZE = 10  # files per commit

    batches = [
        to_upload[i : i + BATCH_SIZE]
        for i in range(0, len(to_upload), BATCH_SIZE)
    ]

    for batch_idx, batch in enumerate(batches):
        print(f"Batch {batch_idx + 1}/{len(batches)}: {[f.name for f in batch]}")

        operations = [
            CommitOperationAdd(
                path_in_repo = f.name,
                path_or_fileobj = str(f),
            )
            for f in batch
        ]

        api.create_commit(
            repo_id    = repo_id,
            repo_type  = "dataset",
            operations = operations,
            commit_message = (
                f"Upload cache batch {batch_idx + 1}/{len(batches)}"
            ),
        )
        print(f"  ✓ Batch {batch_idx + 1} committed\n")

    print(f"\n✓ Upload complete → https://huggingface.co/datasets/{repo_id}")
    print(
        "\nYour friend can download with:\n"
        f"  from huggingface_hub import snapshot_download\n"
        f"  snapshot_download(\n"
        f"      repo_id     = '{repo_id}',\n"
        f"      repo_type   = 'dataset',\n"
        f"      local_dir   = '/gector-data/cache',\n"
        f"      max_workers = 8,\n"
        f"  )\n"
    )


# ── Modal entrypoint ──────────────────────────────────────────────────────────

@app.function(
    image   = image,
    cpu     = 2,
    memory  = 4096,
    volumes = {MOUNT: volume},
    timeout = 7200,   # 2 hours — large cache upload
    secrets = [modal.Secret.from_name("huggingface-secret")],
)
def upload_on_modal(
    repo_id:   str,
    cache_dir: str  = f"{MOUNT}/cache",
    private:   bool = True,
):
    upload_cache(
        repo_id   = repo_id,
        cache_dir = cache_dir,
        private   = private,
        token     = os.environ["HF_TOKEN"],
    )


@app.local_entrypoint()
def main(
    repo_id:    str,
    cache_dir:  str  = "",        # empty = use Modal volume path
    private:    bool = True,
    run_local:  bool = False,     # True = run on your machine, False = run on Modal
):
    """
    Upload GECToR cache to HuggingFace.

    Run on Modal (recommended — reads directly from volume):
        modal run upload_cache_to_hub.py --repo_id yourname/gector-cache

    Run locally (if you have the cache on disk):
        modal run upload_cache_to_hub.py \\
            --repo_id yourname/gector-cache \\
            --cache_dir /path/to/cache \\
            --run_local
    """
    if run_local:
        if not cache_dir:
            raise ValueError("--cache_dir required when --run_local is set")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("Set HF_TOKEN environment variable")
        upload_cache(
            repo_id   = repo_id,
            cache_dir = cache_dir,
            private   = private,
            token     = token,
        )
    else:
        resolved_cache = cache_dir or f"{MOUNT}/cache"
        upload_on_modal.spawn(
            repo_id   = repo_id,
            cache_dir = resolved_cache,
            private   = private,
        )