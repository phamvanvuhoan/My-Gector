"""
beam_train.py — Three-stage GECToR training on Beam Cloud (RTX 4090).

Prerequisites
─────────────
1. Install & configure the Beam SDK locally:
       pip install beam-client
       beam configure default --token <YOUR_TOKEN>

2. Create the persistent volume (once):
       beam volume create gector-data

3. Upload your preprocessed data to the volume:
       # Upload files to the volume root — Beam preserves the local filename.
       # run setup_volume_beam.py to do all of this automatically.
       beam cp data/output_vocabulary  beam://gector-data
       beam cp data/stage1.train       beam://gector-data
       beam cp data/stage1.dev         beam://gector-data
       # … repeat for stage2 / stage3 files

4. Store your W&B API key as a Beam secret (once):
       beam secret create WANDB_API_KEY <your_key>

Usage
─────
# Preprocess (CPU-only, fast):
    python beam_train.py preprocess

# Train stage 1:
    python beam_train.py stage1

# Train stage 2 (auto-resumes from stage 1 checkpoint):
    python beam_train.py stage2

# Train stage 3:
    python beam_train.py stage3

# Download best checkpoint from stage 3:
    python beam_train.py download --stage 3 --which best --local-dir ./my_model

Optional overrides (any stage):
    python beam_train.py stage1 --model-id bert-base-cased --batch-size 512
"""

import argparse
import os
import sys
from pathlib import Path

from beam import Image, Volume, function, task_queue

# ── Shared volume / paths ──────────────────────────────────────────────────────

VOLUME_NAME = "gector-data"
MOUNT       = "./gector-data"          # Beam mounts at a relative path inside the container

# Files are uploaded to the volume root via `beam cp <file> beam://gector-data`
# so they land directly at MOUNT/<filename> — there is no extra data/ subfolder.
DATA        = MOUNT
VOCAB_DIR   = f"{MOUNT}/output_vocabulary"
SAVE_BASE   = f"{MOUNT}/checkpoints"
HF_CACHE    = f"{MOUNT}/hf_cache"

# ── Per-stage hyperparameters ──────────────────────────────────────────────────

STAGE_CFG = {
    1: dict(
        train_file    = f"{DATA}/stage1.train",
        valid_file    = f"{DATA}/stage1.dev",
        batch_size    = 512,       # RTX 4090 has 24 GB VRAM; tune this down if OOM
        n_cold_epochs = 2,
        n_epochs      = 10,
        save_dir      = f"{SAVE_BASE}/stage1",
    ),
    2: dict(
        train_file    = f"{DATA}/stage2.train",
        valid_file    = f"{DATA}/stage2.dev",
        batch_size    = 256,
        n_cold_epochs = 2,
        n_epochs      = 10,
        save_dir      = f"{SAVE_BASE}/stage2",
    ),
    3: dict(
        train_file    = f"{DATA}/stage3.train",
        valid_file    = f"{DATA}/stage3.dev",
        batch_size    = 256,
        n_cold_epochs = 0,
        n_epochs      = 10,
        save_dir      = f"{SAVE_BASE}/stage3",
    ),
}

# ── Container image (built once, cached by Beam) ───────────────────────────────

gector_image = (
    Image(python_version="python3.11")
    .add_commands(["apt-get update -y", "apt-get install -y git"])
    .add_python_packages([
        "torch>=2.6.0",
        "transformers>=4.49.0",
        "accelerate>=1.3.0",
        "huggingface-hub>=0.28.1",
        "python-levenshtein>=0.26.1",
        "wandb",
        "psutil",   # for RAM logging in preprocess_all
    ])
    .add_commands([
        # Install your fork of gector
        "pip install --no-cache-dir --force-reinstall git+https://github.com/phamvanvuhoan/My-Gector.git@beam-version"
    ])
)

# Beam Volume object — attached to every function below
gector_volume = Volume(name=VOLUME_NAME, mount_path=MOUNT)

# ── Preprocessing (CPU-only) ───────────────────────────────────────────────────

@task_queue(
    image   = gector_image,
    cpu     = 8,
    memory  = "64Gi",   # stage1 has 8.8M sentences; 32 Gi OOM-killed at ~26%
    volumes = [gector_volume],
    timeout = 14400,    # 4 h — sharded processing of stage1 takes longer
    secrets = ["WANDB_API_KEY"],
)
def preprocess_all(
    model_id:   str = "roberta-base",
    max_len:    int = 80,
    shard_size: int = 500_000,   # sentences per shard; lower to 200k if still OOM
):
    """
    Tokenise and cache all stage datasets on CPU using sharded saves.

    Instead of holding all 8.8M tokenised tensors in RAM before writing,
    we process `shard_size` sentences at a time and flush each shard to
    the volume immediately.  Peak RAM is bounded to ~1 shard at a time.
    A mid-run crash is safe — already-saved shards are skipped on retry.
    """
    import os
    import psutil
    from transformers import AutoTokenizer
    from gector import preprocess_dataset

    def log_mem(tag: str):
        mb = psutil.Process().memory_info().rss / 1e6
        print(f"  [RAM] {tag}: {mb:.0f} MB")

    os.makedirs(HF_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, add_prefix_space=True
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["$START"]})

    stages = [
        (f"{DATA}/stage1.train", "stage1 train"),
        (f"{DATA}/stage1.dev",   "stage1 dev"),
        (f"{DATA}/stage2.train", "stage2 train"),
        (f"{DATA}/stage2.dev",   "stage2 dev"),
        (f"{DATA}/stage3.train", "stage3 train"),
        (f"{DATA}/stage3.dev",   "stage3 dev"),
    ]

    for file_path, name in stages:
        if not os.path.exists(file_path):
            print(f"SKIP {name}: not found at {file_path}")
            continue
        print(f"\n{'='*50}\nProcessing {name} …\n{'='*50}")
        log_mem("before")
        preprocess_dataset(
            input_file = file_path,
            tokenizer  = tokenizer,
            max_length = max_len,
            shard_size = shard_size,
        )
        log_mem("after")
        print(f"✓ {name} cached")

    print("\n✓ All preprocessing done.")


# ── Core training function (RTX 4090) ─────────────────────────────────────────

@task_queue(
    image   = gector_image,
    gpu     = "RTX4090",
    cpu     = 8,
    memory  = "32Gi",
    volumes = [gector_volume],
    timeout = 86400,             # 24 h — stage 1 can be long; set -1 to disable
    secrets = ["WANDB_API_KEY"],
)
def train_stage(
    stage:              int,
    model_id:           str   = "roberta-base",
    restore_dir:        str   = None,
    lr:                 float = 1e-5,
    cold_lr:            float = 1e-3,
    max_len:            int   = 80,
    n_max_labels:       int   = 5000,
    accumulation:       int   = 1,
    label_smoothing:    float = 0.0,
    num_warmup_steps:   int   = 500,
    lr_scheduler_type:  str   = "constant",
    seed:               int   = 10,
    wandb_project:      str   = "gector",
    wandb_run_name:     str   = None,
):
    """Launch one training stage inside the Beam container."""
    import subprocess
    import sys
    import os

    os.makedirs(HF_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = HF_CACHE

    cfg      = STAGE_CFG[stage]
    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # Auto-restore from previous stage's best checkpoint
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

    run_name = wandb_run_name or f"stage{stage}_{model_id}"

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--mixed_precision", "fp16",
        "train.py",
        "--train_file",        cfg["train_file"],
        "--valid_file",        cfg["valid_file"],
        "--save_dir",          save_dir,
        "--batch_size",        str(cfg["batch_size"]),
        "--n_cold_epochs",     str(cfg["n_cold_epochs"]),
        "--n_epochs",          str(cfg["n_epochs"]),
        "--ckpt_steps",        "500",
        "--ckpt_limit",        "2",
        "--resume_ckpt",       "auto",
        "--lr",                str(lr),
        "--cold_lr",           str(cold_lr),
        "--max_len",           str(max_len),
        "--n_max_labels",      str(n_max_labels),
        "--accumulation",      str(accumulation),
        "--label_smoothing",   str(label_smoothing),
        "--num_warmup_steps",  str(num_warmup_steps),
        "--lr_scheduler_type", lr_scheduler_type,
        "--seed",              str(seed),
        "--restore_vocab_official", VOCAB_DIR,
        "--wandb_project",     wandb_project,
        "--wandb_run_name",    run_name,
    ]

    if restore_dir:
        cmd += ["--restore_dir", restore_dir]
    else:
        cmd += ["--model_id", model_id]

    print(f"\n=== Stage {stage} command ===")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True, env=os.environ.copy())

    print(f"\n✓ Stage {stage} complete. Checkpoints saved to {save_dir}")
    return save_dir


# ── Local entrypoints (called from your machine) ───────────────────────────────

def _maybe_override_batch(stage: int, batch_size: int):
    if batch_size > 0:
        STAGE_CFG[stage]["batch_size"] = batch_size


def _dispatch(fn, kwargs: dict, detach: bool, label: str):
    """
    Dispatch a task_queue function.

    detach=True  → .put() and return immediately (terminal-safe).
    detach=False → .put() then block by calling .remote() so output streams
                   to your terminal as before.

    task_queue functions use .put() to enqueue; the task runs on Beam
    regardless of whether your local process stays alive.
    """
    if detach:
        task = fn.put(**kwargs)
        print(f"✓ {label} dispatched (detached).")
        print(f"  Task ID : {task.id}")
        print(f"  Monitor : beam logs {task.id}")
        print(f"  Cancel  : beam cancel {task.id}")
    else:
        # .put() enqueues, then we wait by calling the synchronous path
        # task_queue doesn't block on .put(), so use .remote() for blocking
        fn.remote(**kwargs)


def cmd_preprocess(args):
    kwargs = dict(
        model_id   = args.model_id,
        max_len    = args.max_len,
        shard_size = args.shard_size,
    )
    _dispatch(preprocess_all, kwargs, args.detach, "Preprocessing")


def cmd_stage(args, stage: int):
    _maybe_override_batch(stage, args.batch_size)
    kwargs = dict(
        stage       = stage,
        model_id    = args.model_id,
        restore_dir = args.restore_dir,
        lr          = args.lr,
        seed        = args.seed,
    )
    _dispatch(train_stage, kwargs, args.detach, f"Stage {stage} training")


def cmd_download(args):
    import subprocess
    remote = f"beam://{VOLUME_NAME}/checkpoints/stage{args.stage}/{args.which}"
    local  = Path(args.local_dir)
    local.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {remote} → {args.local_dir} …")
    subprocess.run(["beam", "cp", remote, str(local)], check=True)
    print("✓ Download complete.")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        description="GECToR training on Beam Cloud"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # shared --detach flag added to every subcommand via parents
    detach_parser = argparse.ArgumentParser(add_help=False)
    detach_parser.add_argument(
        "--detach", action="store_true",
        help=(
            "Fire-and-forget: dispatch the job and exit immediately. "
            "The cloud process keeps running after you close your terminal. "
            "Use 'beam logs <task-id>' to follow progress."
        )
    )

    # preprocess
    p = sub.add_parser("preprocess",
                       help="Tokenise & cache all datasets (CPU)",
                       parents=[detach_parser])
    p.add_argument("--model-id",   default="roberta-base", dest="model_id")
    p.add_argument("--max-len",    type=int, default=80, dest="max_len")
    p.add_argument("--shard-size", type=int, default=500_000, dest="shard_size",
                   help="Sentences per cache shard (lower if still OOM, e.g. 200000)")

    # stage1 / stage2 / stage3
    for s in (1, 2, 3):
        p = sub.add_parser(f"stage{s}", help=f"Train stage {s}",
                           parents=[detach_parser])
        p.add_argument("--model-id",    default="roberta-base", dest="model_id")
        p.add_argument("--restore-dir", default=None, dest="restore_dir",
                       help="Override checkpoint path (default: auto from previous stage)")
        p.add_argument("--batch-size",  type=int, default=0, dest="batch_size",
                       help="Override default batch size (0 = use stage default)")
        p.add_argument("--lr",          type=float, default=1e-5)
        p.add_argument("--seed",        type=int, default=10)

    # download
    p = sub.add_parser("download", help="Download a checkpoint to your local machine")
    p.add_argument("--stage",     type=int, default=3)
    p.add_argument("--which",     default="best", choices=["best", "last"])
    p.add_argument("--local-dir", default="outputs/beam_checkpoint", dest="local_dir")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    cmd    = args.command

    if cmd == "preprocess":
        cmd_preprocess(args)
    elif cmd == "stage1":
        cmd_stage(args, 1)
    elif cmd == "stage2":
        cmd_stage(args, 2)
    elif cmd == "stage3":
        cmd_stage(args, 3)
    elif cmd == "download":
        cmd_download(args)
    else:
        parser.print_help()
        sys.exit(1)