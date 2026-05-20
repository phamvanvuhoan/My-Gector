"""
modal_train.py — GECToR three-stage training on Modal.

Pipeline
--------
1. Upload raw data to the volume (done outside this file).
2. Run preprocess_all once — CPU-heavy, no GPU needed.
3. Run run_stage{1,2,3} — GPU training, resumes automatically after preemption.

Usage
-----
    modal run modal_train.py::preprocess
    modal run modal_train.py::run_stage1
    modal run modal_train.py::run_stage2
    modal run modal_train.py::run_stage3
    modal run modal_train.py::download_checkpoint --stage 3
"""

import os
from pathlib import Path

import modal

# ── Infrastructure ────────────────────────────────────────────────────────────

app    = modal.App("gector-train")
volume = modal.Volume.from_name("gector-data", create_if_missing=False)
MOUNT  = "/gector-data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .env({"FORCE_REBUILD": "2024-05-36"})
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.49.0",
        "accelerate>=1.3.0",
        "huggingface-hub>=0.28.1",
        "python-levenshtein>=0.26.1",
        "wandb",
    )
    .run_commands(
        "pip install --no-cache-dir git+https://github.com/phamvanvuhoan/My-Gector.git"
    )
    .add_local_file("train.py", "/root/train.py")
)

# ── Shared paths ──────────────────────────────────────────────────────────────

DATA      = f"{MOUNT}/data"
VOCAB_DIR = f"{DATA}/output_vocabulary"
CACHE_DIR = f"{MOUNT}/cache"
SAVE_BASE = f"{MOUNT}/checkpoints"
HF_CACHE  = f"{MOUNT}/hf_cache"

# Per-stage training configuration
STAGE_CFG = {
    1: dict(
        train_file    = f"{DATA}/stage1.train",
        valid_file    = f"{DATA}/stage1.dev",
        batch_size    = 512,   # cold epochs
        warm_batch_size = 512,   # warm epochs
        n_cold_epochs = 2,
        n_epochs      = 5,
        save_dir      = f"{SAVE_BASE}/stage1",
    ),
    2: dict(
        train_file    = f"{DATA}/stage2.train",
        valid_file    = f"{DATA}/stage2.dev",
        batch_size    = 512,
        warm_batch_size = 256,   # warm epochs
        n_cold_epochs = 1,
        n_epochs      = 10,
        save_dir      = f"{SAVE_BASE}/stage2",
    ),
    3: dict(
        train_file    = f"{DATA}/stage3.train",
        valid_file    = f"{DATA}/stage3.dev",
        batch_size    = 256,
        warm_batch_size = 128,   # warm epochs
        n_cold_epochs = 0,
        n_epochs      = 10,
        save_dir      = f"{SAVE_BASE}/stage3",
    ),
}


# ── Preprocessing (CPU) ───────────────────────────────────────────────────────

@app.function(
    image   = image,
    cpu     = 6,
    memory  = 36864,      # 36 GB — stage1 has 8.8M sentences
    volumes = {MOUNT: volume},
    timeout = 14400,      # 4 hours
)
def preprocess_all(
    model_id:  str = "roberta-base",
    max_len:   int = 80,
):
    """
    Tokenize and cache all stage datasets on CPU.
    Must be run once before any training stage.

    For each input file this writes:
        <cache>/<file>.cache_<hash>.input_ids.mmap
        <cache>/<file>.cache_<hash>.attention_mask.mmap
        <cache>/<file>.cache_<hash>.word_masks.mmap
        <cache>/<file>.cache_<hash>.labels.mmap
        <cache>/<file>.cache_<hash>.d_labels.mmap
        <cache>/<file>.cache_<hash>.meta.pt

    Resumes automatically after preemption via per-file progress files.
    Files whose meta.pt already exists are skipped.
    """
    import os
    from transformers import AutoTokenizer
    from gector.dataset import build_cache
    from gector.vocab import load_vocab_from_official

    os.environ["GECTOR_CACHE_DIR"] = CACHE_DIR
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(HF_CACHE,  exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, add_prefix_space=True
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["$START"]})

    label2id, d_label2id = load_vocab_from_official(VOCAB_DIR)

    files = [
        f"{DATA}/stage1.train", f"{DATA}/stage1.dev",
        f"{DATA}/stage2.train", f"{DATA}/stage2.dev",
        f"{DATA}/stage3.train", f"{DATA}/stage3.dev",
    ]

    for file_path in files:
        if not os.path.exists(file_path):
            print(f"SKIP (not found): {file_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print(f"{'='*60}")

        build_cache(
            input_file    = file_path,
            tokenizer     = tokenizer,
            label2id      = label2id,
            d_label2id    = d_label2id,
            max_length    = max_len,
            commit_fn     = lambda: volume.commit(),
        )
        # Commit after each file so partial progress survives preemption.
        volume.commit()

    print("\n✓ All preprocessing complete.")


@app.local_entrypoint()
def preprocess(
    model_id: str = "roberta-base",
    max_len:  int = 80,
):
    preprocess_all.spawn(model_id=model_id, max_len=max_len)


# ── Training (GPU) ────────────────────────────────────────────────────────────

@app.function(
    image   = image,
    gpu     = "A100-40GB",
    cpu     = 4,
    memory  = 8192,
    volumes = {MOUNT: volume},
    timeout = 86400,   # 24 h
    secrets = [modal.Secret.from_name("wandb-secret")],
    retries = modal.Retries(
        max_retries         = 2,
        backoff_coefficient = 1.0,
        initial_delay       = 5.0,
    ),
)
def train_stage(
    stage:             int,
    model_id:          str   = "roberta-base",
    restore_dir:       str   = None, # change to f"{SAVE_BASE}/stage1/last" if want to change max_weight at the beginning of epoch
    max_weight:        float = 3.0,
    lr:                float = 1e-5,
    cold_lr:           float = 1e-3,
    max_len:           int   = 80,
    n_max_labels:      int   = 5000,
    accumulation:      int   = 1,
    label_smoothing:   float = 0.0,
    num_warmup_steps:  int   = 200,
    lr_scheduler_type: str   = "constant",
    seed:              int   = 10,
):
    """
    Launch one training stage inside a Modal GPU container.
    Resumes automatically from the latest step checkpoint after preemption.
    Stage 2 and 3 default to restoring from the previous stage's best checkpoint.
    """
    import subprocess
    import sys

    os.environ["HF_HOME"]          = HF_CACHE
    os.environ["GECTOR_CACHE_DIR"] = CACHE_DIR

    cfg      = STAGE_CFG[stage]
    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Default restore: previous stage's best checkpoint
    if restore_dir is None and stage > 1:
        prev_best = STAGE_CFG[stage - 1]["save_dir"] + "/best"
        if Path(prev_best).exists():
            restore_dir = prev_best
            print(f"Stage {stage}: restoring weights from {restore_dir}")
        else:
            print(
                f"WARNING: expected checkpoint at {prev_best} but not found. "
                "Training from scratch with --model_id."
            )

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--mixed_precision", "fp16", # A100s support bf16, which is faster and has higher effective batch size than fp16 (change back to "fp16" if using other GPUs)
        "train.py",
        "--train_file",          cfg["train_file"],
        "--valid_file",          cfg["valid_file"],
        "--save_dir",            save_dir,
        "--batch_size",          str(cfg["batch_size"]),
        "--warm_batch_size",     str(cfg["warm_batch_size"]),
        "--n_cold_epochs",       str(cfg["n_cold_epochs"]),
        "--n_epochs",            str(cfg["n_epochs"]),
        "--lr",                  str(lr),
        "--cold_lr",             str(cold_lr),
        "--max_len",             str(max_len),
        "--n_max_labels",        str(n_max_labels),
        "--accumulation",        str(accumulation),
        "--label_smoothing",     str(label_smoothing),
        "--num_warmup_steps",    str(num_warmup_steps),
        "--lr_scheduler_type",   lr_scheduler_type,
        "--seed",                str(seed),
        "--resume_ckpt",         "auto",
        "--ckpt_steps",          "1000",
        "--ckpt_limit",          "2",
        "--restore_vocab_official", VOCAB_DIR,
        "--wandb_project",       "gector",
        "--wandb_run_name",      f"stage{stage}_{model_id}_v3",
        "--max_weight",          str(max_weight),
    ]

    if restore_dir:
        cmd += ["--restore_dir", restore_dir]
    else:
        cmd += ["--model_id", model_id]

    print(f"\n=== Stage {stage} command ===")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True, env=os.environ.copy())

    volume.commit()
    print(f"\n✓ Stage {stage} complete. Checkpoints at {save_dir}")
    return save_dir


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=8192,
    volumes={MOUNT: volume},
    timeout=180,
)
def test():
    from transformers import AutoTokenizer
    from gector import GECToR, predict, load_verb_dict
    import torch

    model = GECToR.from_pretrained("/gector-data/checkpoints/stage1/last").eval()
    tokenizer = AutoTokenizer.from_pretrained("/gector-data/checkpoints/stage1/last")
    encode, decode = load_verb_dict("/gector-data/data/verb-form-vocab.txt")

    if torch.cuda.is_available():
        model.cuda()

    srcs = [
        "This are wrong sentences",
        "He go to school yesterday",
        "I have went to the store",
        "She don't knows the answer",
        "There is many people here",
    ]

    corrected = predict(
        model, tokenizer, srcs, encode, decode,
        keep_confidence = 0.0,
        min_error_prob  = 0.0,
        n_iteration     = 1,
    )

    for src, cor in zip(srcs, corrected):
        print(f"SRC: {src}")
        print(f"COR: {cor}")
        print()

# ── Per-stage entrypoints ─────────────────────────────────────────────────────

@app.local_entrypoint()
def run_stage1(
    model_id:   str   = "roberta-base",
    batch_size: int   = 0,
    lr:         float = 1e-5,
    num_warmup_steps: int = 500,
    seed:       int   = 10,
):
    """Train stage 1 (large synthetic corpus)."""
    _maybe_override_batch(1, batch_size)
    train_stage.spawn(stage=1, model_id=model_id, lr=lr, num_warmup_steps=num_warmup_steps, seed=seed)


@app.local_entrypoint()
def run_stage2(
    restore_dir: str   = None,
    batch_size:  int   = 0,
    lr:          float = 1e-5,
    num_warmup_steps: int = 200,
    seed:        int   = 10,
):
    """Train stage 2 (BEA19 corpus), resumes from stage 1 by default."""
    _maybe_override_batch(2, batch_size)
    train_stage.spawn(stage=2, restore_dir=restore_dir, lr=lr, num_warmup_steps=num_warmup_steps, seed=seed)


@app.local_entrypoint()
def run_stage3(
    restore_dir: str   = None,
    batch_size:  int   = 0,
    lr:          float = 5e-6,
    num_warmup_steps: int = 100,
    seed:        int   = 10,
):
    """Train stage 3 (W&I+LOCNESS fine-tune), resumes from stage 2 by default."""
    _maybe_override_batch(3, batch_size)
    train_stage.spawn(stage=3, restore_dir=restore_dir, lr=lr, num_warmup_steps=num_warmup_steps, seed=seed)


@app.local_entrypoint()
def quick_test():
    """Quick test to verify the training container can load the model and run inference."""
    test.remote()
# ── Download ──────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def download_checkpoint(
    stage:     int = 3,
    which:     str = "best",
    local_dir: str = "outputs/modal_checkpoint",
):
    """
    Copy a trained checkpoint from the Modal Volume to your local machine.

        modal run modal_train.py::download_checkpoint --stage 3
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


# ── Internal helpers ──────────────────────────────────────────────────────────

def _maybe_override_batch(stage: int, batch_size: int) -> None:
    if batch_size > 0:
        STAGE_CFG[stage]["batch_size"] = batch_size