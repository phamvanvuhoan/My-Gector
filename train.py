import argparse
from transformers import AutoTokenizer, get_scheduler
from gector import (
    GECToR,
    GECToRConfig,
    load_dataset,
    build_vocab,
    load_vocab_from_config,
    load_vocab_from_official
)
from gector.vocab import compute_class_weights
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
import os
from tqdm import tqdm
import json
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import numpy as np
import random
from collections import OrderedDict
from gector.utils import has_args_add_pooling
from pathlib import Path
import wandb


def solve_model_id(model_id):
    if model_id == 'deberta-base':
        return 'microsoft/deberta-base'
    elif model_id == 'deberta-large':
        return 'microsoft/deberta-large'
    else:
        return model_id

def _prune_checkpoints(save_dir: str, limit: int):
    ckpts = sorted(
        Path(save_dir).glob("checkpoint_*"),
        key=os.path.getmtime
    )
    for old in ckpts[:-limit]:
        import shutil
        shutil.rmtree(old)
        print(f"  Pruned old checkpoint: {old}")

def train(
    model,
    loader,
    optimizer,
    lr_scheduler,
    accelerator,
    epoch,
    step_scheduler,
    global_step,          # <-- add
    ckpt_steps,  # <-- add
    save_dir,             # <-- add
    ckpt_limit, # <-- add
    resume_step=0,        # <-- add (only non-zero on first resumed epoch)
):
    log = {
        'loss': 0,
        'accuracy': 0,
        'accuracy_d': 0
    }
    n_batches = 0  # track actual batches run (skip doesn't count)
    model.train()
    pbar = tqdm(loader, total=len(loader), disable=not accelerator.is_main_process)
    for step, batch in enumerate(pbar):

        # Skip steps already done before preemption
        if step < resume_step:
            continue

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            if step_scheduler:
                lr_scheduler.step()

        global_step += 1
        n_batches += 1
        log['loss'] += loss.item()
        log['accuracy'] += outputs.accuracy.item()
        log['accuracy_d'] += outputs.accuracy_d.item()

        if accelerator.is_main_process:
            pbar.set_description(f'[Epoch {epoch}] [TRAIN]')
            pbar.set_postfix(OrderedDict(
                loss=loss.item(),
                accuracy=outputs.accuracy.item(),
                accuracy_d=outputs.accuracy_d.item(),
                lr=optimizer.param_groups[0]['lr']
            ))
            # ── W&B step log ─────────────────────────────────────
            if wandb.run is not None and global_step % 50 ==0:
                wandb.log({
                    'train/loss':       loss.item(),
                    'train/accuracy':   outputs.accuracy.item(),
                    'train/accuracy_d': outputs.accuracy_d.item(),
                    'train/lr':         optimizer.param_groups[0]['lr'],
                    'train/epoch':      epoch,
                }, step=global_step)

        # ── Step checkpoint ───────────────────────────────────────
        if global_step % ckpt_steps == 0:
            accelerator.wait_for_everyone()
            ckpt_path = os.path.join(save_dir, f"checkpoint_{global_step}")  # <-- define it here
            accelerator.save_state(ckpt_path)   # all processes must call this
            if accelerator.is_main_process:
                _prune_checkpoints(save_dir, ckpt_limit)
                print(f"  [step {global_step}] checkpoint saved → {ckpt_path}")

    n_batches = max(n_batches, 1)  # avoid div-by-zero if all steps were skipped
    return {k: v / n_batches for k, v in log.items()}, global_step

@torch.no_grad()
def valid(
    model,
    loader,
    accelerator,
    epoch
):
    log = {
        'loss': 0,
        'accuracy': 0,
        'accuracy_d': 0
    }
    model.eval()
    pbar = tqdm(loader, total=len(loader), disable=not accelerator.is_main_process)
    for batch in pbar:
        outputs = model(**batch)
        log['loss'] += outputs.loss.item()
        log['accuracy'] += outputs.accuracy.item()
        log['accuracy_d'] += outputs.accuracy_d.item()
        if accelerator.is_main_process:
            pbar.set_description(f'[Epoch {epoch}] [VALID]')
            pbar.set_postfix(OrderedDict(
                loss=outputs.loss.item(),
                accuracy=outputs.accuracy.item(),
                accuracy_d=outputs.accuracy_d.item(),
            ))
    return {k:v/len(loader) for k,v in log.items()}

def main(args):
    # To easily specify the model_id 
    args.model_id = solve_model_id(args.model_id)
    print('Start ...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = True

    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation,
                              project_dir=args.save_dir,
    )
    import wandb
    # only log from main process to avoid duplicate runs
    if accelerator.is_main_process and args.wandb_project:
        wandb.init(
            project  = args.wandb_project,
            name     = args.wandb_run_name,
            config   = vars(args),         # logs all hyperparams automatically
            resume   = "allow",            # resumes the same run after preemption
            id       = args.wandb_run_name # stable id so preemption resume works
        )

    if args.restore_dir is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.restore_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id,
            add_prefix_space=True
        )
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['$START']}
    )

    print('Loading datasets...')
    dataset_args = {
        'input_file': args.train_file,
        'tokenizer': tokenizer,
        'delimeter': args.delimeter,
        'additional_delimeter': args.additional_delimeter,
        'max_length': args.max_len
    }
    train_dataset = load_dataset(**dataset_args)
    dataset_args['input_file'] = args.valid_file
    valid_dataset = load_dataset(**dataset_args)
    if args.restore_dir is not None:
        # If you specify path or id to --restore_dir, the model loads weights and vocab.
        model = GECToR.from_pretrained(args.restore_dir)
    else:
        # Otherwise, the model will be trained from scratch.
        if args.restore_vocab is not None:
            # But you can use existing vocab.
            label2id, d_label2id = load_vocab_from_config(args.restore_vocab)
        elif args.restore_vocab_official is not None:
            label2id, d_label2id = load_vocab_from_official(args.restore_vocab_official)
        else:
            print('Builing vocab...')
            label2id, d_label2id = build_vocab(
                train_dataset,
                n_max_labels=args.n_max_labels,
                n_max_d_labels=2
            )
        gector_config = GECToRConfig(
            model_id=args.model_id,
            label2id=label2id,
            id2label={v: k for k, v in label2id.items()},
            d_label2id=d_label2id,
            p_dropout=args.p_dropout,
            max_length=args.max_len,
            label_smoothing=args.label_smoothing,
            has_add_pooling_layer=has_args_add_pooling(args.model_id)
        )
        model = GECToR(config=gector_config)
    train_dataset.append_vocab(
        model.config.label2id,
        model.config.d_label2id
    )
    valid_dataset.append_vocab(
        model.config.label2id,
        model.config.d_label2id
    )
    label_weights = compute_class_weights(
    train_dataset,
    model.config.label2id,          # or gector_config.label2id if building fresh
    strategy='sqrt_inverse_freq',   # gentler than raw inverse for GEC
    max_weight=10.0
    )
    # Store in config so it's saved with the checkpoint
    model.config.label_weights = label_weights.tolist()
    # Inject into the already-constructed loss_fn
    model.loss_fn = CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
        weight=label_weights
    )
    print('# instances of train:', len(train_dataset))
    print('# instances of valid:', len(valid_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,        # <-- add
        pin_memory=True,      # <-- add, faster CPU→GPU transfer
        prefetch_factor=4,    # <-- prefetch 2 batches per worker
        persistent_workers=True,  # <-- keep workers alive between epochs
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.accumulation,
        num_training_steps=len(train_loader) * (args.n_epochs - args.n_cold_epochs) // args.accumulation,
    )
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, valid_loader, lr_scheduler
    )
    # ── Resume logic ──────────────────────────────────────────────────────────────
    resume_step  = 0
    resume_epoch = 0
    global_step  = 0

    resume_path = args.resume_ckpt
    if resume_path == "auto":
        ckpt_dirs = [
            d for d in Path(args.save_dir).glob("checkpoint_*") if d.is_dir()
        ]
        if ckpt_dirs:
            resume_path = str(max(ckpt_dirs, key=os.path.getmtime))
            print(f"Auto-resuming from {resume_path}")
        else:
            resume_path = None
            print("No checkpoint found, starting from scratch.")

    if resume_path and Path(resume_path).exists():
        accelerator.load_state(resume_path)
        try:
            global_step  = int(Path(resume_path).name.split("_")[-1])
            steps_per_epoch = len(train_loader)
            resume_epoch = global_step // steps_per_epoch
            resume_step  = global_step % steps_per_epoch
            print(f"Resumed at global_step={global_step} "
                f"(epoch {resume_epoch}, step {resume_step})")
        except ValueError:
            pass
    # ─────────────────────────────────────────────────────────────────────────────

    path_to_best = os.path.join(args.save_dir, 'best')
    path_to_last = os.path.join(args.save_dir, 'last')
    os.makedirs(path_to_best, exist_ok=True)
    os.makedirs(path_to_last, exist_ok=True)
    tokenizer.save_pretrained(path_to_best)
    tokenizer.save_pretrained(path_to_last)
    max_acc = -1
    print('Start training...')
    def set_lr(optimizer, lr):
        for param in optimizer.param_groups:
            param['lr'] = lr
    logs = {'argparse': args.__dict__}
    for e in range(resume_epoch, args.n_epochs):
        accelerator.wait_for_everyone()
        if isinstance(model, DistributedDataParallel):
            module = model.module
        else:
            module = model
        step_scheduler = True
        if e < args.n_cold_epochs:
            module.tune_bert(False)
            set_lr(optimizer, args.cold_lr)
            step_scheduler = False
        elif e == args.n_cold_epochs:
            module.tune_bert(True)
            set_lr(optimizer, args.lr)
        else:
            pass
        print(f'=== Epoch {e} ===')
        train_log, global_step = train(
            model,
            train_loader,
            optimizer,
            lr_scheduler,
            accelerator,
            e,
            step_scheduler,
            global_step=global_step,                            # <-- add
            ckpt_steps=args.ckpt_steps,       # <-- add
            save_dir=args.save_dir,                             # <-- add
            ckpt_limit=args.ckpt_limit, # <-- add
            resume_step=resume_step if e == resume_epoch else 0,  # <-- add
        )
        valid_log = valid(
            model,
            valid_loader,
            accelerator,
            e
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if valid_log['accuracy'] > max_acc:
                accelerator.unwrap_model(model).save_pretrained(path_to_best)
                max_acc = valid_log['accuracy']
                valid_log['message'] = 'The best checkpoint has been updated.'
            accelerator.unwrap_model(model).save_pretrained(path_to_last)
            logs[f'Epoch {e}'] = {
                'train_log': train_log,
                'valid_log': valid_log
            }
            with open(os.path.join(args.save_dir, 'log.json'), 'w') as f:
                json.dump(logs, f, indent=2)
            # ── W&B epoch log ─────────────────────────────────────────────
            if args.wandb_project:
                wandb.log({
                    'epoch/train_loss':       train_log['loss'],
                    'epoch/train_accuracy':   train_log['accuracy'],
                    'epoch/train_accuracy_d': train_log['accuracy_d'],
                    'epoch/valid_loss':       valid_log['loss'],
                    'epoch/valid_accuracy':   valid_log['accuracy'],
                    'epoch/valid_accuracy_d': valid_log['accuracy_d'],
                    'epoch/is_best':          valid_log.get('message') is not None,
                }, step=global_step)
    if accelerator.is_main_process and args.wandb_project:
        wandb.finish()
    print('finish')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--valid_file', required=True)
    parser.add_argument('--model_id', default='bert-base-cased')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--delimeter', default='SEPL|||SEPR')
    parser.add_argument('--additional_delimeter', default='SEPL__SEPR')
    parser.add_argument('--restore_dir')
    parser.add_argument('--restore_vocab')
    parser.add_argument('--restore_vocab_official')
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--n_max_labels', type=int, default=5000)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--p_dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--cold_lr', type=float, default=1e-3)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--n_cold_epochs', type=int, default=2)
    parser.add_argument('--num_warmup_steps', type=int, default=500)
    parser.add_argument(
        "--lr_scheduler_type",
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    # In get_parser(), add:
    parser.add_argument('--resume_from_checkpoint', default=None,
        help='Path to an Accelerate checkpoint dir to resume from. '
            'Pass "auto" to auto-find the latest in save_dir.')
    parser.add_argument('--checkpointing_steps', type=int, default=1000,
        help='Save an Accelerate checkpoint every N steps.')
    parser.add_argument('--checkpoints_total_limit', type=int, default=2,
        help='Keep only the N most recent step checkpoints to save disk space.')
    parser.add_argument('--resume_ckpt', default=None,
        help='"auto" to find latest checkpoint, or explicit path.')
    parser.add_argument('--ckpt_steps', type=int, default=500)
    parser.add_argument('--ckpt_limit', type=int, default=2)
    parser.add_argument('--wandb_project',  default=None,
        help='W&B project name. Omit to disable wandb.')
    parser.add_argument('--wandb_run_name', default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)

# bert-base-cased roberta-base deberta-base xlnet-base-cased
# bert-large-cased roberta-large deberta-large xlnet-large-cased