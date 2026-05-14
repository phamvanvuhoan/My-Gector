"""
dataset.py

Assumes all cache files have been written by preprocess_all() in modal_train.py
before training begins. load_dataset() is a thin loader — no tokenization,
no vocab application, no fallback logic.

Cache layout for each input file:
    <cache_dir>/<filename>.cache_<hash>.input_ids.mmap       int64 (n, max_length)
    <cache_dir>/<filename>.cache_<hash>.attention_mask.mmap  int64 (n, max_length)
    <cache_dir>/<filename>.cache_<hash>.word_masks.mmap      int64 (n, max_length)
    <cache_dir>/<filename>.cache_<hash>.labels.mmap          int64 (n, max_length)
    <cache_dir>/<filename>.cache_<hash>.d_labels.mmap        int64 (n, max_length)
    <cache_dir>/<filename>.cache_<hash>.meta.pt              {n: int}
"""

import os
import hashlib
import pickle
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

# Can be overridden by setting GECTOR_CACHE_DIR before import.
CACHE_DIR = os.environ.get("GECTOR_CACHE_DIR", "/gector-data/cache")


# ── Cache path ────────────────────────────────────────────────────────────────

def _cache_path(input_file: str, tokenizer: PreTrainedTokenizer, max_length: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = f"{input_file}_{tokenizer.name_or_path}_{max_length}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    filename = os.path.basename(input_file)
    return os.path.join(CACHE_DIR, f"{filename}.cache_{h}")


# ── Dataset ───────────────────────────────────────────────────────────────────

class GECToRDataset:
    """
    Thin wrapper around five memory-mapped arrays produced by preprocess_all().
    __getitem__ copies one row out of each mmap — zero in-memory tensors at rest,
    so worker processes don't fight over shared memory.
    """
    def __init__(
        self,
        input_ids:       np.ndarray,   # mmap (n, max_length) int64
        attention_masks: np.ndarray,   # mmap (n, max_length) int64
        word_masks:      np.ndarray,   # mmap (n, max_length) int64
        labels:          np.ndarray,   # mmap (n, max_length) int64
        d_labels:        np.ndarray,   # mmap (n, max_length) int64
    ):
        assert len(input_ids) == len(labels), "mmap length mismatch"
        self.input_ids       = input_ids
        self.attention_masks = attention_masks
        self.word_masks      = word_masks
        self.labels          = labels
        self.d_labels        = d_labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        # .copy() turns the mmap row into a regular numpy array before
        # converting to tensor — avoids multiprocessing mmap issues.
        return {
            "input_ids":      torch.from_numpy(self.input_ids[idx].copy()),
            "attention_mask": torch.from_numpy(self.attention_masks[idx].copy()),
            "word_masks":     torch.from_numpy(self.word_masks[idx].copy()),
            "labels":         torch.from_numpy(self.labels[idx].copy()),
            "d_labels":       torch.from_numpy(self.d_labels[idx].copy()),
        }


# ── Loader (training path) ────────────────────────────────────────────────────

def load_dataset(
    input_file: str,
    tokenizer:  PreTrainedTokenizer,
    max_length: int = 128,
) -> GECToRDataset:
    """
    Load a fully pre-processed dataset from cache.
    Raises FileNotFoundError if preprocessing has not been run yet.
    """
    cache_file = _cache_path(input_file, tokenizer, max_length)

    required = {
        "input_ids":      cache_file + ".input_ids.mmap",
        "attention_mask": cache_file + ".attention_mask.mmap",
        "word_masks":     cache_file + ".word_masks.mmap",
        "labels":         cache_file + ".labels.mmap",
        "d_labels":       cache_file + ".d_labels.mmap",
        "meta":           cache_file + ".meta.pt",
    }

    missing = [name for name, path in required.items() if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"Cache incomplete for {input_file}. "
            f"Missing: {missing}. "
            f"Run preprocess_all() first."
        )

    meta = torch.load(required["meta"], weights_only=True)
    n    = meta["n"]

    def _mmap(path: str) -> np.ndarray:
        return np.memmap(path, dtype="int64", mode="r", shape=(n, max_length))

    return GECToRDataset(
        input_ids       = _mmap(required["input_ids"]),
        attention_masks = _mmap(required["attention_mask"]),
        word_masks      = _mmap(required["word_masks"]),
        labels          = _mmap(required["labels"]),
        d_labels        = _mmap(required["d_labels"]),
    )


# ── Preprocessing (called only from modal_train.preprocess_all) ───────────────

def load_gector_format(
    input_file:           str,
    delimeter:            str = "SEPL|||SEPR",
    additional_delimeter: str = "SEPL__SEPR",
):
    """Parse a GECToR-preprocessed file into parallel lists of tokens + labels."""
    srcs, word_level_labels = [], []
    with open(input_file) as f:
        for line in f:
            src    = [x.split(delimeter)[0] for x in line.split()]
            labels = [x.split(delimeter)[1] for x in line.split()]
            labels = [l.split(additional_delimeter)[0] for l in labels]
            srcs.append(src)
            word_level_labels.append(labels)
    return srcs, word_level_labels


def build_cache(
    input_file:   str,
    tokenizer:    PreTrainedTokenizer,
    label2id:     dict,
    d_label2id:   dict,
    max_length:   int   = 128,
    batch_size:   int   = 10_000,
    commit_fn     = None,   # callable[[], None] — e.g. volume.commit() on Modal
    keep_label:   str   = "$KEEP",
    pad_token:    str   = "<PAD>",
    correct_label: str  = "$CORRECT",
    incorrect_label: str = "$INCORRECT",
) -> str:
    """
    Tokenize input_file, align labels to subwords, apply vocab, and write
    six cache files. Resumes automatically after preemption via a progress file.

    Returns the cache_file prefix so the caller can verify or commit.
    """
    cache_file = _cache_path(input_file, tokenizer, max_length)
    progress_file = cache_file + ".progress.pkl"
    meta_file     = cache_file + ".meta.pt"

    # Already complete — nothing to do.
    if os.path.exists(meta_file):
        meta = torch.load(meta_file, weights_only=True)
        print(f"✓ Already cached: {os.path.basename(input_file)} (n={meta['n']:,})")
        return cache_file

    # ── Parse source file ────────────────────────────────────────────────────
    print(f"Parsing {input_file} ...")
    srcs, word_level_labels = load_gector_format(input_file)
    n = len(srcs)
    print(f"  {n:,} sentences")

    # ── Pre-allocate mmaps ───────────────────────────────────────────────────
    def _open_mmap(suffix: str, mode: str) -> np.ndarray:
        path = cache_file + suffix
        m = "r+" if (mode == "r+" and os.path.exists(path)) else "w+"
        return np.memmap(path, dtype="int64", mode=m, shape=(n, max_length))

    input_ids_mm      = _open_mmap(".input_ids.mmap",      "r+")
    attention_mask_mm = _open_mmap(".attention_mask.mmap",  "r+")
    word_masks_mm     = _open_mmap(".word_masks.mmap",      "r+")
    labels_mm         = _open_mmap(".labels.mmap",          "r+")
    d_labels_mm       = _open_mmap(".d_labels.mmap",        "r+")

    # ── Resume state ─────────────────────────────────────────────────────────
    if os.path.exists(progress_file):
        with open(progress_file, "rb") as f:
            progress = pickle.load(f)
        start_batch = progress["last_completed_batch"] + 1
        print(f"  Resuming from batch {start_batch} ({start_batch * batch_size:,}/{n:,} sentences)")
    else:
        start_batch = 0

    oov_id   = label2id["<OOV>"]
    d_pad_id = d_label2id["<PAD>"]

    # ── Process in batches ───────────────────────────────────────────────────
    batches = list(range(0, n, batch_size))
    for batch_idx, i in enumerate(tqdm(batches, desc=os.path.basename(input_file))):
        if batch_idx < start_batch:
            continue

        batch_srcs   = srcs[i : i + batch_size]
        batch_labels = word_level_labels[i : i + batch_size]
        batch_n      = len(batch_srcs)

        encoded = tokenizer(
            batch_srcs,
            max_length=max_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            is_split_into_words=True,
        )

        input_ids_mm[i : i + batch_n]      = encoded["input_ids"].numpy()
        attention_mask_mm[i : i + batch_n] = encoded["attention_mask"].numpy()

        for j, wlabels in enumerate(batch_labels):
            label_row   = []
            d_label_row = []
            wmask       = []
            prev_word   = None

            for word_idx in encoded.word_ids(j):
                if word_idx is None:
                    label_row.append(label2id[pad_token])       # <PAD> id
                    d_label_row.append(d_pad_id)
                    wmask.append(0)
                elif word_idx != prev_word:
                    raw = wlabels[word_idx]
                    label_row.append(label2id.get(raw, oov_id))
                    d_label_row.append(
                        d_label2id[incorrect_label] if raw != keep_label
                        else d_label2id[correct_label]
                    )
                    wmask.append(1)
                else:
                    label_row.append(label2id[pad_token])
                    d_label_row.append(d_pad_id)
                    wmask.append(0)
                prev_word = word_idx

            def _pad(arr, pad_val):
                arr = np.array(arr, dtype="int64")
                if len(arr) < max_length:
                    arr = np.pad(arr, (0, max_length - len(arr)),
                                 constant_values=pad_val)
                return arr[:max_length]

            labels_mm[i + j]   = _pad(label_row,   label2id[pad_token])
            d_labels_mm[i + j] = _pad(d_label_row, d_pad_id)
            word_masks_mm[i + j] = _pad(wmask,      0)

        # ── Flush and checkpoint every 5 batches ─────────────────────────
        if batch_idx % 5 == 0:
            for mm in (input_ids_mm, attention_mask_mm, word_masks_mm,
                       labels_mm, d_labels_mm):
                mm.flush()
            with open(progress_file, "wb") as f:
                pickle.dump({"last_completed_batch": batch_idx}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            if commit_fn is not None:
                commit_fn()

    # ── Final flush + atomic meta write ──────────────────────────────────────
    for mm in (input_ids_mm, attention_mask_mm, word_masks_mm,
               labels_mm, d_labels_mm):
        mm.flush()

    tmp = meta_file + ".tmp"
    torch.save({"n": n}, tmp)
    os.replace(tmp, meta_file)   # atomic on Linux

    if os.path.exists(progress_file):
        os.remove(progress_file)

    if commit_fn is not None:
        commit_fn()

    print(f"✓ Cache complete: {os.path.basename(input_file)} (n={n:,})")
    return cache_file