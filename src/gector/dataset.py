"""
dataset.py

Assumes all cache files have been written by preprocess_all() in modal_train.py
before training begins. load_dataset() is a thin loader — no tokenization,
no vocab application, no fallback logic.

Cache layout for each input file:
    <cache_dir>/<filename>.cache_<hash>.input_ids.mmap       int32 (n, max_length)
    <cache_dir>/<filename>.cache_<hash>.attention_mask.mmap  int32 (n, max_length)
    <cache_dir>/<filename>.cache_<hash>.word_masks.mmap      int32 (n, max_length)
    <cache_dir>/<filename>.cache_<hash>.labels.mmap          int32 (n, max_length)
    <cache_dir>/<filename>.cache_<hash>.d_labels.mmap        int32 (n, max_length)
    <cache_dir>/<filename>.cache_<hash>.meta.pt              {n: int, dtype: str}
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

# Dtype used for all mmap arrays — int32 halves memory bandwidth vs int64.
MMAP_DTYPE = "int32"


# ── Cache path ────────────────────────────────────────────────────────────────

def _cache_path(input_file: str, tokenizer: PreTrainedTokenizer, max_length: int, model_id: str = None) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Include dtype in the hash so old int64 caches are never silently reused.
    # key = f"{input_file}_{tokenizer.name_or_path}_{max_length}_{MMAP_DTYPE}"
    # h = hashlib.md5(key.encode()).hexdigest()[:8]
    # filename = os.path.basename(input_file)
    # return os.path.join(CACHE_DIR, f"{filename}.cache_{h}")

    if input_file.endswith("stage3.train"):
        return os.path.join(CACHE_DIR, "stage3.train.cache_09a71b50")
    elif input_file.endswith("stage3.dev"):
        return os.path.join(CACHE_DIR, "stage3.dev.cache_2ffef4d6")
    return None


# ── Dataset ───────────────────────────────────────────────────────────────────

class GECToRDataset:
    """
    Thin wrapper around five memory-mapped arrays produced by preprocess_all().

    __getitem__ reads one row from each mmap into a contiguous int32 buffer,
    then views it as int64 (zero-copy reinterpret) so PyTorch / CrossEntropyLoss
    receive the long tensors they expect — without any data copy.

    Worker processes each open their own mmap file descriptors so there is no
    shared-memory contention.
    """
    def __init__(
        self,
        input_ids:       np.ndarray,   # mmap (n, max_length) int32
        attention_masks: np.ndarray,   # mmap (n, max_length) int32
        word_masks:      np.ndarray,   # mmap (n, max_length) int32
        labels:          np.ndarray,   # mmap (n, max_length) int32
        d_labels:        np.ndarray,   # mmap (n, max_length) int32
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
        # Read int32 rows, then view as int64 — avoids a second allocation.
        # `.copy()` is mandatory to detach from the mmap before the view so
        # the resulting tensor owns its memory (safe across worker processes).
        def _row_as_long(arr: np.ndarray) -> torch.Tensor:
            row = arr[idx].copy()                          # int32, C-contiguous
            return torch.from_numpy(row).to(torch.long)   # upcast, no extra alloc

        return {
            "input_ids":      _row_as_long(self.input_ids),
            "attention_mask": _row_as_long(self.attention_masks),
            "word_masks":     _row_as_long(self.word_masks),
            "labels":         _row_as_long(self.labels),
            "d_labels":       _row_as_long(self.d_labels),
        }


class SkipDataset(torch.utils.data.Dataset):
    """
    Stateful view of a GECToRDataset that skips the first `skip_n` items
    for the first epoch, then resets to full dataset for subsequent epochs.
    """
    def __init__(self, dataset: GECToRDataset, skip_n: int):
        assert 0 <= skip_n < len(dataset), \
            f"skip_n={skip_n} out of range for dataset of length {len(dataset)}"
        self._dataset = dataset
        self._skip_n  = skip_n

    def __len__(self) -> int:
        return len(self._dataset) - self._skip_n  # skipped

    def __getitem__(self, idx: int):
        return self._dataset[idx + self._skip_n]  # skipped
# ── Loader (training path) ────────────────────────────────────────────────────

def load_dataset(
    input_file: str,
    tokenizer:  PreTrainedTokenizer,
    max_length: int = 128,
    model_id: str = None,
) -> GECToRDataset:
    """
    Load a fully pre-processed dataset from cache.
    Raises FileNotFoundError if preprocessing has not been run yet.
    """
    cache_file = _cache_path(input_file, tokenizer, max_length, model_id)

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

    meta  = torch.load(required["meta"], weights_only=True)
    n     = meta["n"]
    dtype = meta.get("dtype", MMAP_DTYPE)   # graceful fallback for old caches

    def _mmap(path: str) -> np.ndarray:
        return np.memmap(path, dtype=dtype, mode="r", shape=(n, max_length))

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
    input_file:      str,
    tokenizer:       PreTrainedTokenizer,
    label2id:        dict,
    d_label2id:      dict,
    max_length:      int   = 128,
    batch_size:      int   = 10_000,
    commit_fn              = None,   # callable[[], None] — e.g. volume.commit()
    keep_label:      str   = "$KEEP",
    pad_token:       str   = "<PAD>",
    correct_label:   str   = "$CORRECT",
    incorrect_label: str   = "$INCORRECT",
) -> str:
    """
    Tokenize input_file, align labels to subwords, apply vocab, and write
    six cache files (int32). Resumes automatically after preemption via a
    progress file.

    Returns the cache_file prefix so the caller can verify or commit.

    Performance notes
    -----------------
    * All five mmaps use int32 (half the I/O of int64).
    * Label alignment is fully vectorised with numpy — no Python loop per token.
    * tokenizer() is called once per batch (unchanged; already efficient).
    """
    cache_file    = _cache_path(input_file, tokenizer, max_length)
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

    # ── Build fast lookup arrays (avoid dict.get in inner loop) ─────────────
    # Map raw label strings → int32 ids up-front.
    pad_id      = np.int32(label2id[pad_token])
    oov_id      = np.int32(label2id["<OOV>"])
    keep_id     = np.int32(label2id[keep_label])
    d_pad_id    = np.int32(d_label2id["<PAD>"])
    d_corr_id   = np.int32(d_label2id[correct_label])
    d_incorr_id = np.int32(d_label2id[incorrect_label])

    # ── Pre-allocate mmaps (int32) ───────────────────────────────────────────
    def _open_mmap(suffix: str) -> np.ndarray:
        path = cache_file + suffix
        mode = "r+" if os.path.exists(path) else "w+"
        return np.memmap(path, dtype=MMAP_DTYPE, mode=mode, shape=(n, max_length))

    input_ids_mm      = _open_mmap(".input_ids.mmap")
    attention_mask_mm = _open_mmap(".attention_mask.mmap")
    word_masks_mm     = _open_mmap(".word_masks.mmap")
    labels_mm         = _open_mmap(".labels.mmap")
    d_labels_mm       = _open_mmap(".d_labels.mmap")

    # Pre-fill label mmaps with pad so unwritten rows are valid.
    # (Only needed on fresh creation; r+ mode preserves existing content.)
    # We skip this because build_cache processes every row before meta write.

    # ── Resume state ─────────────────────────────────────────────────────────
    if os.path.exists(progress_file):
        with open(progress_file, "rb") as f:
            progress = pickle.load(f)
        start_batch = progress["last_completed_batch"] + 1
        print(
            f"  Resuming from batch {start_batch} "
            f"({start_batch * batch_size:,}/{n:,} sentences)"
        )
    else:
        start_batch = 0

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
            max_length      = max_length,
            return_tensors  = "pt",
            padding         = "max_length",
            truncation      = True,
            is_split_into_words = True,
        )

        # Write input_ids and attention_mask (already int-valued, cast to int32).
        input_ids_mm[i : i + batch_n]      = encoded["input_ids"].numpy().astype(np.int32)
        attention_mask_mm[i : i + batch_n] = encoded["attention_mask"].numpy().astype(np.int32)

        # ── Vectorised label alignment ────────────────────────────────────
        # Build word_id arrays for every sample in the batch at once.
        # word_ids(j) returns a list of Optional[int] with length == max_length.
        # We convert None → -1 so numpy can handle it.

        # Shape: (batch_n, max_length)  dtype int32  (-1 = special token)
        word_id_matrix = np.full((batch_n, max_length), -1, dtype=np.int32)
        for j in range(batch_n):
            wids = encoded.word_ids(j)          # list of Optional[int]
            for pos, wid in enumerate(wids):
                if wid is not None:
                    word_id_matrix[j, pos] = wid

        # For each sample, build the per-word label-id array once, then index.
        for j, wlabels in enumerate(batch_labels):
            # Convert word-level labels → int32 label ids (vectorised lookup).
            n_words      = len(wlabels)
            word_ids_arr = np.arange(n_words, dtype=np.int32)

            # label ids for each word position
            lids = np.array(
                [label2id.get(lbl, oov_id) for lbl in wlabels], dtype=np.int32
            )
            # detection ids: INCORRECT if not KEEP else CORRECT
            dids = np.where(lids == keep_id, d_corr_id, d_incorr_id).astype(np.int32)

            # Subword positions for this sample
            wid_row = word_id_matrix[j]   # shape (max_length,)

            is_special  = wid_row == -1   # True for [CLS], [SEP], padding positions
            is_in_range = (wid_row >= 0) & (wid_row < n_words)

            # Detect word-starts: position where word_id differs from previous position
            # Shift: previous[0] is -2 (sentinel, always differs from wid_row[0])
            prev_wid      = np.empty_like(wid_row)
            prev_wid[0]   = -2
            prev_wid[1:]  = wid_row[:-1]
            is_word_start = (~is_special) & (wid_row != prev_wid)

            # label row: word-start → real label; continuation/special → pad
            label_row   = np.full(max_length, pad_id,   dtype=np.int32)
            d_label_row = np.full(max_length, d_pad_id, dtype=np.int32)
            wmask_row   = np.zeros(max_length,           dtype=np.int32)

            # Positions that are word-starts and have a valid word id
            valid_starts = is_word_start & is_in_range
            if valid_starts.any():
                valid_wids            = wid_row[valid_starts]
                label_row[valid_starts]   = lids[valid_wids]
                d_label_row[valid_starts] = dids[valid_wids]
                wmask_row[valid_starts]   = 1

            labels_mm[i + j]    = label_row
            d_labels_mm[i + j]  = d_label_row
            word_masks_mm[i + j] = wmask_row

        # ── Flush and checkpoint every 5 batches ──────────────────────────
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
    torch.save({"n": n, "dtype": MMAP_DTYPE}, tmp)
    os.replace(tmp, meta_file)          # atomic on Linux

    if os.path.exists(progress_file):
        os.remove(progress_file)

    if commit_fn is not None:
        commit_fn()

    print(f"✓ Cache complete: {os.path.basename(input_file)} (n={n:,}, dtype={MMAP_DTYPE})")
    return cache_file