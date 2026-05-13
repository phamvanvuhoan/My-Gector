"""
modal_train.py — Three-stage GECToR training on Modal.

Requires data already uploaded via setup_volume_modal.py.

Usage:
    modal run modal_train.py::run_stage1
    modal run modal_train.py::run_stage2 --model-id gotutiyan/gector-roberta-base-5k
    modal run modal_train.py::run_stage3

Optional overrides (any stage):
    modal run modal_train.py::run_stage1 --model-id bert-base-cased --batch-size 128
"""

import os
import modal
from pathlib import Path

# ── Infrastructure ─────────────────────────────────────────────────────────────

app    = modal.App("gector-train")
volume = modal.Volume.from_name("gector-data", create_if_missing=False)
MOUNT  = "/gector-data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .env({"FORCE_REBUILD": "2024-05-14"})
    .apt_install("git")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.49.0",
        "accelerate>=1.3.0",
        "huggingface-hub>=0.28.1",
        "python-levenshtein>=0.26.1",
        "wandb",
    )
    .run_commands(
        #hope this help me rebuild image v1
        "pip install --no-cache-dir git+https://github.com/phamvanvuhoan/My-Gector.git"
    )
    .add_local_file("train.py", "/root/train.py")   # <-- add this
)

GPU = "A10G"  # swap to A10G() for cheaper runs; A100 recommended for stage1

# ── Shared paths inside the volume ────────────────────────────────────────────

DATA      = f"{MOUNT}/data"
VOCAB_DIR = f"{DATA}/output_vocabulary"
SAVE_BASE = f"{MOUNT}/checkpoints"

STAGE_CFG = {
    1: dict(
        train_file  = f"{DATA}/stage1.train",
        valid_file  = f"{DATA}/stage1.dev",
        batch_size  = 512,
        n_cold_epochs = 2,
        n_epochs    = 10,
        save_dir    = f"{SAVE_BASE}/stage1",
    ),
    2: dict(
        train_file  = f"{DATA}/stage2.train",
        valid_file  = f"{DATA}/stage2.dev",
        batch_size  = 512,
        n_cold_epochs = 2,
        n_epochs    = 10,
        save_dir    = f"{SAVE_BASE}/stage2",
    ),
    3: dict(
        train_file  = f"{DATA}/stage3.train",
        valid_file  = f"{DATA}/stage3.dev",
        batch_size  = 512,
        n_cold_epochs = 0,
        n_epochs    = 10,
        save_dir    = f"{SAVE_BASE}/stage3",
    ),
}

# add this to modal_train.py

@app.function(
    image   = image,
    cpu     = 4,           # more CPUs = faster tokenization
    memory  = 49152,       # 48 GB — stage1 is 8.8M sentences
    volumes = {MOUNT: volume},
    timeout = 7200,        # 2 hours should be enough for all stages
)
def preprocess_all(
    model_id:   str = "roberta-base",
    max_len:    int = 80,
):
    """
    Tokenize and cache all stage datasets on CPU.
    Run this ONCE before any training stage.

    Usage:
        modal run modal_train.py::preprocess_all
        modal run modal_train.py::preprocess_all --model-id bert-base-cased
    """
    import os
    from transformers import AutoTokenizer
    from gector import load_dataset

    cache_dir = f"{MOUNT}/cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["GECTOR_CACHE_DIR"] = cache_dir

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        add_prefix_space=True
    )
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['$START']}
    )

    stages = [
        (f"{MOUNT}/data/stage1.train", "stage1 train"),
        (f"{MOUNT}/data/stage1.dev",   "stage1 dev"),
        (f"{MOUNT}/data/stage2.train", "stage2 train"),
        (f"{MOUNT}/data/stage2.dev",   "stage2 dev"),
        (f"{MOUNT}/data/stage3.train", "stage3 train"),
        (f"{MOUNT}/data/stage3.dev",   "stage3 dev"),
    ]

    for file_path, name in stages:
        if not os.path.exists(file_path):
            print(f"SKIP {name}: file not found at {file_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Processing {name} ...")
        print(f"{'='*50}")

        load_dataset(
            input_file = file_path,
            tokenizer  = tokenizer,
            max_length = max_len,
            use_cache  = True,
            commit_fn  = lambda: volume.commit(),   # <-- add
        )

        # flush to volume after each file so partial progress is saved
        # if this function gets interrupted
        volume.commit()
        print(f"✓ {name} cached and committed to volume")

    print("\n✓ All preprocessing done. Ready to train.")

@app.function(
    image   = image,
    cpu     = 8,
    memory  = 65536,
    volumes = {MOUNT: volume},
    timeout = 3600,
)
def apply_vocab_to_cache(
    model_id: str = "roberta-base",
    max_len:  int = 80,
):
    """
    One-time step: load existing cache meta files, apply vocab,
    save back. After this, training skips append_vocab entirely.
    """
    import torch
    import numpy as np
    import gc
    from pathlib import Path
    from transformers import AutoTokenizer
    from gector import load_vocab_from_official

    cache_dir = f"{MOUNT}/cache"
    label2id, d_label2id = load_vocab_from_official(
        f"{MOUNT}/data/output_vocabulary"
    )
    oov_id   = label2id['<OOV>']
    d_pad_id = d_label2id['<PAD>']

    meta_files = sorted(Path(cache_dir).glob("*.meta.pt"))
    if not meta_files:
        print("No meta files found in cache dir")
        return

    for meta_file in meta_files:
        print(f"\nProcessing {meta_file.name} ...")
        meta = torch.load(str(meta_file), weights_only=False)

        if meta.get('vocab_applied', False):
            print("  Already applied, skipping")
            continue

        print(f"  Applying vocab to {meta['n']:,} sentences...")
        labels   = meta['labels']
        d_labels = meta['d_labels']

        # convert string labels to ints
        print("  Converting labels...")
        labels_int = [
            [label2id.get(l, oov_id) for l in sent]
            for sent in labels
        ]
        print("  Converting d_labels...")
        d_labels_int = [
            [d_label2id.get(l, d_pad_id) for l in sent]
            for sent in d_labels
        ]

        # overwrite meta file with vocab-applied labels
        print("  Saving updated meta...")
        torch.save({
            'n':            meta['n'],
            'srcs':         meta['srcs'],
            'labels':       labels_int,
            'd_labels':     d_labels_int,
            'vocab_applied': True,          # <-- flag so training skips append_vocab
        }, str(meta_file))

        del labels, d_labels, labels_int, d_labels_int
        gc.collect()

        volume.commit()
        print(f"  ✓ Done and committed")

    print("\n✓ All cache files updated. Training will now skip append_vocab.")


@app.local_entrypoint()
def apply_vocab():
    apply_vocab_to_cache.spawn()

@app.function(
    image   = image,
    cpu     = 8,
    memory  = 16384,    # low memory needed — converts chunk by chunk
    volumes = {MOUNT: volume},
    timeout = 3600,
)
def convert_mmap_to_int64():
    """
    One-time conversion of existing int32 mmap files to int64.
    Much faster than reprocessing from scratch.
    """
    import numpy as np
    from pathlib import Path

    cache_dir = f"{MOUNT}/cache"
    mmap_files = sorted(Path(cache_dir).glob("*.mmap"))

    if not mmap_files:
        print("No mmap files found")
        return

    for mmap_file in mmap_files:
        print(f"\nConverting {mmap_file.name} ...")

        # read shape from existing int32 file
        old = np.memmap(str(mmap_file), dtype='int32', mode='r')
        total_elements = old.shape[0]

        # read meta to get (n, max_length) shape
        # infer from file size: total_elements = n * max_length
        # we know max_length=80 from training config
        max_length = 80
        n = total_elements // max_length
        old = np.memmap(str(mmap_file), dtype='int32', mode='r', shape=(n, max_length))

        size_gb = mmap_file.stat().st_size / 1e9
        print(f"  Shape: ({n}, {max_length}), size: {size_gb:.2f} GB")

        # write to temp file as int64
        tmp_file = str(mmap_file) + '.int64.tmp'
        new = np.memmap(tmp_file, dtype='int64', mode='w+', shape=(n, max_length))

        # convert in chunks to avoid RAM spike
        chunk = 50000
        for i in range(0, n, chunk):
            new[i:i+chunk] = old[i:i+chunk].astype('int64')
            if i % 500000 == 0:
                new.flush()
                print(f"  {i}/{n} rows converted...")

        new.flush()
        del old, new

        # replace original with converted file
        import os, shutil
        os.replace(tmp_file, str(mmap_file))
        print(f"  ✓ Converted {mmap_file.name}")

        volume.commit()

    print("\n✓ All mmap files converted to int64")


@app.local_entrypoint()
def convert_cache():
    convert_mmap_to_int64.remote()

# Verify cache integrity before training — run this to check for any issues with the cached files that could cause training to fail. This is especially useful if you had an interruption during preprocessing or if you want to sanity-check the cache before launching a long training run.
@app.function(
    image   = image,
    cpu     = 4,
    memory  = 16384,
    volumes = {MOUNT: volume},
    timeout = 600,
)
def verify_cache():
    import numpy as np
    import torch
    import os
    from pathlib import Path

    cache_dir = f"{MOUNT}/cache"
    issues    = []

    cache_files = list(Path(cache_dir).glob("*.meta.pt"))
    if not cache_files:
        print("No cache files found!")
        return

    for meta_file in sorted(cache_files):
        base = str(meta_file).replace('.meta.pt', '')
        name = meta_file.name.replace('.meta.pt', '')
        print(f"\nChecking {name} ...")

        # 1. check all expected files exist
        expected = [
            meta_file,
            base + '.input_ids.mmap',
            base + '.attention_mask.mmap',
            base + '.word_masks.mmap',
        ]
        for f in expected:
            if not os.path.exists(f):
                issues.append(f"MISSING FILE: {f}")
                print(f"  ✗ Missing: {f}")
            else:
                size_gb = os.path.getsize(f) / 1e9
                print(f"  ✓ Exists:  {f} ({size_gb:.2f} GB)")

        # 2. load meta and check contents
        try:
            meta = torch.load(str(meta_file))
            n         = meta['n']
            n_labels  = len(meta['labels'])
            n_dlabels = len(meta['d_labels'])
            n_srcs    = len(meta['srcs'])
            print(f"  n={n}, srcs={n_srcs}, labels={n_labels}, d_labels={n_dlabels}")

            if not (n == n_labels == n_dlabels == n_srcs):
                issues.append(
                    f"SIZE MISMATCH in {name}: "
                    f"n={n} srcs={n_srcs} labels={n_labels} d_labels={n_dlabels}"
                )
                print(f"  ✗ Size mismatch!")
            else:
                print(f"  ✓ Meta sizes consistent")
        except Exception as e:
            issues.append(f"META LOAD ERROR in {name}: {e}")
            print(f"  ✗ Failed to load meta: {e}")
            continue

        # 3. check mmap shapes and spot-check values
        try:
            input_ids_mm = np.memmap(
                base + '.input_ids.mmap',
                dtype='int32', mode='r', shape=(n, 80)
            )
            attn_mm = np.memmap(
                base + '.attention_mask.mmap',
                dtype='int32', mode='r', shape=(n, 80)
            )
            wm_mm = np.memmap(
                base + '.word_masks.mmap',
                dtype='int32', mode='r', shape=(n, 80)
            )

            # check first, middle, last rows are not all zeros
            for row_name, idx in [('first', 0), ('middle', n//2), ('last', n-1)]:
                ids_row  = input_ids_mm[idx]
                attn_row = attn_mm[idx]
                if ids_row.sum() == 0:
                    issues.append(f"ALL ZERO input_ids at row {idx} ({row_name}) in {name}")
                    print(f"  ✗ All-zero input_ids at {row_name} row ({idx})")
                elif attn_row.sum() == 0:
                    issues.append(f"ALL ZERO attention_mask at row {idx} ({row_name}) in {name}")
                    print(f"  ✗ All-zero attention_mask at {row_name} row ({idx})")
                else:
                    print(f"  ✓ {row_name} row looks valid (input_ids sum={ids_row.sum()})")

            # check last batch specifically — most likely to be corrupted
            print(f"  Checking last 1000 rows ...")
            last_chunk = input_ids_mm[n-1000:n]
            zero_rows  = (last_chunk.sum(axis=1) == 0).sum()
            if zero_rows > 0:
                issues.append(
                    f"CORRUPT TAIL: {zero_rows} all-zero rows in last 1000 of {name}"
                )
                print(f"  ✗ {zero_rows} zero rows in last 1000 — likely corrupt from heartbeat failure")
            else:
                print(f"  ✓ Last 1000 rows look valid")

        except Exception as e:
            issues.append(f"MMAP ERROR in {name}: {e}")
            print(f"  ✗ mmap failed: {e}")

    # summary
    print(f"\n{'='*50}")
    if issues:
        print(f"FOUND {len(issues)} ISSUE(S):")
        for issue in issues:
            print(f"  ✗ {issue}")
        print("\nRecommendation: clear cache and rerun preprocess for affected files")
    else:
        print("✓ All cache files look valid. Safe to train.")


@app.local_entrypoint()
def verify():
    verify_cache.remote()

# ── Core training function (runs on Modal GPU) ─────────────────────────────────
MAX_RETRIES = 10   # Modal preempts at most this many times before we give up

@app.function(
    image   = image,
    gpu     = GPU,
    cpu     = 8,           # more CPUs = faster data loading
    volumes = {MOUNT: volume},
    timeout = 86400,   # 24 h — stage 1 can be long
    secrets = [modal.Secret.from_name("wandb-secret")],   # <-- add
    retries = modal.Retries(        # Modal-level retry on preemption/OOM
        max_retries    = MAX_RETRIES,
        backoff_coefficient = 1.0,
        initial_delay  = 5.0,
    ),
)
def train_stage(
    stage:         int,
    model_id:      str   = "roberta-base",
    restore_dir:   str   = None,   # path inside volume, e.g. checkpoints/stage1/best
    lr:            float = 1e-5,
    cold_lr:       float = 1e-3,
    max_len:       int   = 80,
    n_max_labels:  int   = 5000,
    accumulation:  int   = 1,
    label_smoothing: float = 0.0,
    num_warmup_steps: int = 500,
    lr_scheduler_type: str = "constant",
    seed:          int   = 10,
):
    """Launches one training stage inside the Modal container."""
    import subprocess, sys, json

    # at the top of train_stage() in modal_train.py, before subprocess.run()
    os.environ["HF_HOME"] = f"{MOUNT}/hf_cache"
    os.makedirs(f"{MOUNT}/hf_cache", exist_ok=True)

    cfg = STAGE_CFG[stage]
    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Stage 2/3: default to resuming from the previous stage's best checkpoint
    if restore_dir is None and stage > 1:
        prev_best = STAGE_CFG[stage - 1]["save_dir"] + "/best"
        if Path(prev_best).exists():
            restore_dir = prev_best
            print(f"Stage {stage}: restoring from {restore_dir}")
        else:
            print(
                f"WARNING: expected checkpoint at {prev_best} but not found. "
                "Training from scratch with --model-id."
            )

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--mixed_precision", "fp16",
        # accelerate will auto-detect the single GPU
        "train.py",                      # assumes train.py is importable via gector install
        "--train_file",       cfg["train_file"],
        "--valid_file",       cfg["valid_file"],
        "--save_dir",         save_dir,
        "--batch_size",       str(cfg["batch_size"]),
        "--n_cold_epochs",    str(cfg["n_cold_epochs"]),
        "--n_epochs",         str(cfg["n_epochs"]),
        "--ckpt_steps",   "500",    # save every 500 steps
        "--ckpt_limit", "2",    # keep last 2 only
        "--resume_ckpt", "auto",  # always try to resume
        "--lr",               str(lr),
        "--cold_lr",          str(cold_lr),
        "--max_len",          str(max_len),
        "--n_max_labels",     str(n_max_labels),
        "--accumulation",     str(accumulation),
        "--label_smoothing",  str(label_smoothing),
        "--num_warmup_steps", str(num_warmup_steps),
        "--lr_scheduler_type", lr_scheduler_type,
        "--seed",             str(seed),
        "--restore_vocab_official", VOCAB_DIR,
    ]

    if restore_dir:
        cmd += ["--restore_dir", restore_dir]
    else:
        cmd += ["--model_id", model_id]

    cmd += [
        "--wandb_project", "gector",
        "--wandb_run_name", f"stage{stage}_{model_id}",
    ]

    print(f"\n=== Stage {stage} command ===")
    print(" ".join(cmd))
    print()

    env = os.environ.copy()
    env["HF_HOME"] = f"{MOUNT}/hf_cache"

    result = subprocess.run(cmd, check=True, env=env)

    # Commit volume writes so next stage / download can see them
    volume.commit()
    print(f"\n✓ Stage {stage} complete. Checkpoints saved to {save_dir}")
    return save_dir


# ── Per-stage entrypoints ──────────────────────────────────────────────────────

@app.local_entrypoint()
def preprocess(
    model_id: str = "roberta-base",
    max_len:  int = 80,
):
    preprocess_all.spawn(model_id=model_id, max_len=max_len)

@app.local_entrypoint()
def clear_cache():
    import subprocess
    clear.spawn()

@app.function(image=image, volumes={MOUNT: volume})
def clear():
    import os

    for root, dirs, files in os.walk(MOUNT):
        for file in files:
            if "cache" in file.lower():
                path = os.path.join(root, file)

                try:
                    os.remove(path)
                    print(f"Deleted {path}")
                except Exception as e:
                    print(f"Failed to delete {path}: {e}")

    volume.commit()
    print("✓ Cache cleared")

@app.local_entrypoint()
def run_stage1(
    model_id:   str   = "roberta-base",
    batch_size: int   = 0,     # 0 = use default from STAGE_CFG
    lr:         float = 1e-5,
    seed:       int   = 10,
):
    """Train stage 1 (large synthetic corpus, cold-start classifier)."""
    _maybe_override_batch(1, batch_size)
    save_dir = train_stage.spawn(
        stage    = 1,
        model_id = model_id,
        lr       = lr,
        seed     = seed,
    )
    print(f"Stage 1 done → {save_dir}")


@app.local_entrypoint()
def run_stage2(
    restore_dir: str   = None,   # defaults to stage1/best
    batch_size:  int   = 0,
    lr:          float = 1e-5,
    seed:        int   = 10,
):
    """Train stage 2 (BEA19 corpus, resume from stage 1)."""
    _maybe_override_batch(2, batch_size)
    save_dir = train_stage.spawn(
        stage       = 2,
        restore_dir = restore_dir,
        lr          = lr,
        seed        = seed,
    )
    print(f"Stage 2 done → {save_dir}")


@app.local_entrypoint()
def run_stage3(
    restore_dir: str   = None,   # defaults to stage2/best
    batch_size:  int   = 0,
    lr:          float = 1e-5,
    seed:        int   = 10,
):
    """Train stage 3 (W&I+LOCNESS fine-tune, no cold epochs)."""
    _maybe_override_batch(3, batch_size)
    save_dir = train_stage.spawn(
        stage       = 3,
        restore_dir = restore_dir,
        lr          = lr,
        seed        = seed,
    )
    print(f"Stage 3 done → {save_dir}")


# ── Download best checkpoint to local disk ────────────────────────────────────

@app.local_entrypoint()
def download_checkpoint(
    stage:     int = 3,
    which:     str = "best",   # "best" or "last"
    local_dir: str = "outputs/modal_checkpoint",
):
    """
    Copy a trained checkpoint from the Modal Volume to your local machine.

    Usage:
        modal run modal_train.py::download_checkpoint --stage 3 --local-dir ./my_model
    """
    remote = f"{SAVE_BASE}/stage{stage}/{which}"
    local  = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {remote} → {local_dir} ...")
    for entry in volume.listdir(remote):
        dest = local / Path(entry.path).name
        with dest.open("wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
        print(f"  {dest}")
    print("✓ Download complete.")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _maybe_override_batch(stage: int, batch_size: int):
    """Allow CLI --batch-size to override the per-stage default."""
    if batch_size > 0:
        STAGE_CFG[stage]["batch_size"] = batch_size