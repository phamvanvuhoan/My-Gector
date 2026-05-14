"""
train.py

Single-GPU or multi-GPU training via Accelerate.
Expects all datasets to be fully preprocessed (run preprocess_all on Modal first).
"""

import argparse
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler

from gector import (
    GECToR,
    GECToRConfig,
    load_dataset,
    load_vocab_from_config,
    load_vocab_from_official,
)
from gector.utils import has_args_add_pooling
from gector.vocab import compute_class_weights


# ── Helpers ───────────────────────────────────────────────────────────────────

def _solve_model_id(model_id: str) -> str:
    shortcuts = {
        "deberta-base":  "microsoft/deberta-base",
        "deberta-large": "microsoft/deberta-large",
    }
    return shortcuts.get(model_id, model_id)


def _set_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def _prune_checkpoints(save_dir: str, limit: int) -> None:
    import shutil
    ckpts = sorted(
        Path(save_dir).glob("checkpoint_*"),
        key=os.path.getmtime,
    )
    for old in ckpts[:-limit]:
        shutil.rmtree(old)
        print(f"  Pruned checkpoint: {old}")


# ── Train / validation loops ──────────────────────────────────────────────────

def train_epoch(
    model,
    loader,
    optimizer,
    lr_scheduler,
    accelerator,
    epoch:        int,
    step_scheduler: bool,
    global_step:  int,
    ckpt_steps:   int,
    ckpt_limit:   int,
    save_dir:     str,
    use_accumulate: bool,
    resume_step:  int = 0,
):
    log = {"loss": 0.0, "accuracy": 0.0, "accuracy_d": 0.0}
    n_batches = 0
    model.train()
    pbar = tqdm(loader, total=len(loader),
                disable=not accelerator.is_main_process)

    for step, batch in enumerate(pbar):
        if step < resume_step:
            continue

        if use_accumulate:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss    = outputs.loss
                optimizer.zero_grad()
                accelerator.backward(loss)
                optimizer.step()
                if step_scheduler:
                    lr_scheduler.step()
        else:
            outputs = model(**batch)
            loss    = outputs.loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

        global_step += 1
        n_batches   += 1
        log["loss"]       += loss.item()
        log["accuracy"]   += outputs.accuracy.item()
        log["accuracy_d"] += outputs.accuracy_d.item()

        if accelerator.is_main_process:
            pbar.set_description(f"[Epoch {epoch}] [TRAIN]")
            if global_step % 10 == 0:
                pbar.set_postfix(OrderedDict(
                    loss=f"{loss.item():.4f}",
                    acc=f"{outputs.accuracy.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                ))
            if wandb.run is not None and global_step % 50 == 0:
                wandb.log({
                    "train/loss":       loss.item(),
                    "train/accuracy":   outputs.accuracy.item(),
                    "train/accuracy_d": outputs.accuracy_d.item(),
                    "train/lr":         optimizer.param_groups[0]["lr"],
                    "train/epoch":      epoch,
                }, step=global_step)

        # ── Step checkpoint ───────────────────────────────────────────────
        if global_step % ckpt_steps == 0:
            accelerator.wait_for_everyone()
            ckpt_path = os.path.join(save_dir, f"checkpoint_{global_step}")
            accelerator.save_state(ckpt_path)
            if accelerator.is_main_process:
                _prune_checkpoints(save_dir, ckpt_limit)
                print(f"  [step {global_step}] checkpoint → {ckpt_path}")

    n_batches = max(n_batches, 1)
    return {k: v / n_batches for k, v in log.items()}, global_step


@torch.no_grad()
def valid_epoch(model, loader, accelerator, epoch: int):
    log = {"loss": 0.0, "accuracy": 0.0, "accuracy_d": 0.0}
    model.eval()
    pbar = tqdm(loader, total=len(loader),
                disable=not accelerator.is_main_process)
    for batch in pbar:
        outputs = model(**batch)
        log["loss"]       += outputs.loss.item()
        log["accuracy"]   += outputs.accuracy.item()
        log["accuracy_d"] += outputs.accuracy_d.item()
        if accelerator.is_main_process:
            pbar.set_description(f"[Epoch {epoch}] [VALID]")
    return {k: v / len(loader) for k, v in log.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    args.model_id = _solve_model_id(args.model_id)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    accelerator    = Accelerator(
        gradient_accumulation_steps=args.accumulation,
        project_dir=args.save_dir,
    )
    use_accumulate = args.accumulation > 1

    if accelerator.is_main_process and args.wandb_project:
        wandb.init(
            project = args.wandb_project,
            name    = args.wandb_run_name,
            config  = vars(args),
            resume  = "allow",
            id      = args.wandb_run_name,
        )

    # ── Tokenizer & vocab ────────────────────────────────────────────────────
    if args.restore_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, add_prefix_space=True
        )
        tokenizer.add_special_tokens({"additional_special_tokens": ["$START"]})

    if args.restore_vocab_official is not None:
        label2id, d_label2id = load_vocab_from_official(args.restore_vocab_official)
    elif args.restore_vocab is not None:
        label2id, d_label2id = load_vocab_from_config(args.restore_vocab)
    else:
        label2id = d_label2id = None  # only valid when restore_dir is set

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("Loading datasets ...")
    train_dataset = load_dataset(args.train_file, tokenizer, args.max_len)
    valid_dataset = load_dataset(args.valid_file, tokenizer, args.max_len)
    print(f"  train: {len(train_dataset):,}  valid: {len(valid_dataset):,}")

    # ── Model ────────────────────────────────────────────────────────────────
    if args.restore_dir is not None:
        model = GECToR.from_pretrained(args.restore_dir)
        label2id   = model.config.label2id
        d_label2id = model.config.d_label2id
    else:
        assert label2id is not None, \
            "Provide --restore_vocab or --restore_vocab_official when not using --restore_dir"

        label_weights = compute_class_weights(
            train_dataset, label2id,
            strategy="sqrt_inverse_freq",
            max_weight=10.0,
        )
        config = GECToRConfig(
            model_id             = args.model_id,
            label2id             = label2id,
            id2label             = {v: k for k, v in label2id.items()},
            d_label2id           = d_label2id,
            p_dropout            = args.p_dropout,
            max_length           = args.max_len,
            label_smoothing      = args.label_smoothing,
            has_add_pooling_layer = has_args_add_pooling(args.model_id),
            label_weights        = label_weights.tolist(),
        )
        model = GECToR(config=config)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size         = args.batch_size,
        shuffle            = True,
        num_workers        = 8,
        pin_memory         = True,
        prefetch_factor    = 2,
        persistent_workers = True,
        multiprocessing_context = "spawn",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size         = args.batch_size,
        shuffle            = False,
        num_workers        = 4,
        pin_memory         = True,
        persistent_workers = True,
        multiprocessing_context = "spawn",
    )

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer    = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name               = args.lr_scheduler_type,
        optimizer          = optimizer,
        num_warmup_steps   = args.num_warmup_steps * args.accumulation,
        num_training_steps = (
            len(train_loader)
            * (args.n_epochs - args.n_cold_epochs)
            // args.accumulation
        ),
    )

    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, lr_scheduler
    )

    # ── Resume ───────────────────────────────────────────────────────────────
    global_step  = 0
    resume_epoch = 0
    resume_step  = 0

    resume_path = args.resume_ckpt
    if resume_path == "auto":
        ckpt_dirs = sorted(
            [d for d in Path(args.save_dir).glob("checkpoint_*") if d.is_dir()],
            key=os.path.getmtime,
        )
        resume_path = str(ckpt_dirs[-1]) if ckpt_dirs else None
        print(f"Auto-resume: {resume_path or 'no checkpoint found, starting fresh'}")

    if resume_path and Path(resume_path).exists():
        accelerator.load_state(resume_path)
        try:
            global_step  = int(Path(resume_path).name.split("_")[-1])
            steps_per_ep = len(train_loader)
            resume_epoch = global_step // steps_per_ep
            resume_step  = global_step  % steps_per_ep
            print(f"Resumed: global_step={global_step}, "
                  f"epoch={resume_epoch}, step_in_epoch={resume_step}")
        except ValueError:
            pass

    # ── Output dirs ───────────────────────────────────────────────────────────
    path_best = os.path.join(args.save_dir, "best")
    path_last = os.path.join(args.save_dir, "last")
    os.makedirs(path_best, exist_ok=True)
    os.makedirs(path_last, exist_ok=True)
    tokenizer.save_pretrained(path_best)
    tokenizer.save_pretrained(path_last)

    # ── Training loop ────────────────────────────────────────────────────────
    max_acc = -1.0
    logs    = {"args": vars(args)}
    print("Starting training ...")

    for epoch in range(resume_epoch, args.n_epochs):
        accelerator.wait_for_everyone()
        module = (model.module
                  if isinstance(model, DistributedDataParallel)
                  else accelerator.unwrap_model(model))

        # Cold / warm phase
        if epoch < args.n_cold_epochs:
            module.tune_bert(False)
            _set_lr(optimizer, args.cold_lr)
            step_scheduler = False
        elif epoch == args.n_cold_epochs:
            module.tune_bert(True)
            _set_lr(optimizer, args.lr)
            step_scheduler = True
        else:
            step_scheduler = True

        print(f"=== Epoch {epoch} ===")
        train_log, global_step = train_epoch(
            model, train_loader, optimizer, lr_scheduler, accelerator,
            epoch        = epoch,
            step_scheduler = step_scheduler,
            global_step  = global_step,
            ckpt_steps   = args.ckpt_steps,
            ckpt_limit   = args.ckpt_limit,
            save_dir     = args.save_dir,
            use_accumulate = use_accumulate,
            resume_step  = resume_step if epoch == resume_epoch else 0,
        )
        valid_log = valid_epoch(model, valid_loader, accelerator, epoch)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            msg = ""
            if valid_log["accuracy"] > max_acc:
                accelerator.unwrap_model(model).save_pretrained(path_best)
                max_acc = valid_log["accuracy"]
                msg = "best checkpoint updated"
            accelerator.unwrap_model(model).save_pretrained(path_last)

            logs[f"epoch_{epoch}"] = {"train": train_log, "valid": valid_log}
            with open(os.path.join(args.save_dir, "log.json"), "w") as f:
                json.dump(logs, f, indent=2)

            if wandb.run is not None:
                wandb.log({
                    "epoch/train_loss":       train_log["loss"],
                    "epoch/train_accuracy":   train_log["accuracy"],
                    "epoch/train_accuracy_d": train_log["accuracy_d"],
                    "epoch/valid_loss":       valid_log["loss"],
                    "epoch/valid_accuracy":   valid_log["accuracy"],
                    "epoch/valid_accuracy_d": valid_log["accuracy_d"],
                    "epoch/is_best":          bool(msg),
                }, step=global_step)

            if msg:
                print(f"  ✓ {msg}")

    if accelerator.is_main_process and args.wandb_project:
        wandb.finish()
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_parser():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train_file",            required=True)
    p.add_argument("--valid_file",            required=True)
    p.add_argument("--save_dir",              required=True)
    # Model
    p.add_argument("--model_id",              default="bert-base-cased")
    p.add_argument("--restore_dir",           default=None)
    p.add_argument("--restore_vocab",         default=None)
    p.add_argument("--restore_vocab_official", default=None)
    p.add_argument("--max_len",               type=int,   default=128)
    p.add_argument("--n_max_labels",          type=int,   default=5000)
    p.add_argument("--p_dropout",             type=float, default=0.0)
    p.add_argument("--label_smoothing",       type=float, default=0.0)
    # Training
    p.add_argument("--batch_size",            type=int,   default=16)
    p.add_argument("--n_epochs",              type=int,   default=10)
    p.add_argument("--n_cold_epochs",         type=int,   default=2)
    p.add_argument("--lr",                    type=float, default=1e-5)
    p.add_argument("--cold_lr",               type=float, default=1e-3)
    p.add_argument("--accumulation",          type=int,   default=1)
    p.add_argument("--seed",                  type=int,   default=10)
    p.add_argument("--num_warmup_steps",      type=int,   default=500)
    p.add_argument("--lr_scheduler_type",     default="constant",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"])
    # Checkpointing & resume
    p.add_argument("--resume_ckpt",           default=None,
        help='"auto" to find latest checkpoint, or explicit path.')
    p.add_argument("--ckpt_steps",            type=int,   default=500,
        help="Save a checkpoint every N optimizer steps.")
    p.add_argument("--ckpt_limit",            type=int,   default=2,
        help="Keep only the N most recent step checkpoints.")
    # Logging
    p.add_argument("--wandb_project",         default=None)
    p.add_argument("--wandb_run_name",        default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(get_parser())