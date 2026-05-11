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
    .apt_install("git")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.49.0",
        "accelerate>=1.3.0",
        "huggingface-hub>=0.28.1",
        "python-levenshtein>=0.26.1",
        "git+https://github.com/phamvanvuhoan/My-Gector.git",
    )
    .add_local_file("train.py", "/root/train.py")   # <-- add this
)

GPU = "A100"  # swap to A10G() for cheaper runs; A100 recommended for stage1

# ── Shared paths inside the volume ────────────────────────────────────────────

DATA      = f"{MOUNT}/data"
VOCAB_DIR = f"{DATA}/output_vocabulary"
SAVE_BASE = f"{MOUNT}/checkpoints"

STAGE_CFG = {
    1: dict(
        train_file  = f"{DATA}/stage1.train",
        valid_file  = f"{DATA}/stage1.dev",
        batch_size  = 256,
        n_cold_epochs = 2,
        n_epochs    = 10,
        save_dir    = f"{SAVE_BASE}/stage1",
    ),
    2: dict(
        train_file  = f"{DATA}/stage2.train",
        valid_file  = f"{DATA}/stage2.dev",
        batch_size  = 128,
        n_cold_epochs = 2,
        n_epochs    = 10,
        save_dir    = f"{SAVE_BASE}/stage2",
    ),
    3: dict(
        train_file  = f"{DATA}/stage3.train",
        valid_file  = f"{DATA}/stage3.dev",
        batch_size  = 128,
        n_cold_epochs = 0,
        n_epochs    = 10,
        save_dir    = f"{SAVE_BASE}/stage3",
    ),
}

# add this to modal_train.py

@app.function(
    image   = image,
    cpu     = 8,           # more CPUs = faster tokenization
    memory  = 32768,       # 32 GB — stage1 is 8.8M sentences
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
        )

        # flush to volume after each file so partial progress is saved
        # if this function gets interrupted
        volume.commit()
        print(f"✓ {name} cached and committed to volume")

    print("\n✓ All preprocessing done. Ready to train.")

# ── Core training function (runs on Modal GPU) ─────────────────────────────────
MAX_RETRIES = 10   # Modal preempts at most this many times before we give up

@app.function(
    image   = image,
    gpu     = GPU,
    volumes = {MOUNT: volume},
    timeout = 86400,   # 24 h — stage 1 can be long
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

    print(f"\n=== Stage {stage} command ===")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, check=True)

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
    preprocess_all.remote(model_id=model_id, max_len=max_len)

@app.local_entrypoint()
def run_stage1(
    model_id:   str   = "roberta-base",
    batch_size: int   = 0,     # 0 = use default from STAGE_CFG
    lr:         float = 1e-5,
    seed:       int   = 10,
):
    """Train stage 1 (large synthetic corpus, cold-start classifier)."""
    _maybe_override_batch(1, batch_size)
    save_dir = train_stage.remote(
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
    save_dir = train_stage.remote(
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
    save_dir = train_stage.remote(
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