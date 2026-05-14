from typing import List, Tuple
from collections import Counter
import torch
from tqdm import tqdm
import os
from transformers import PreTrainedTokenizer
import numpy as np

class GECToRDataset:
    def __init__(
        self,
        srcs,
        d_labels=None,
        labels=None,
        word_masks=None,
        tokenizer=None,
        max_length=128,
        input_ids=None,
        attention_masks=None,
    ):
        self.tokenizer     = tokenizer
        self.srcs          = srcs
        self.max_length    = max_length
        self.label2id      = None
        self.d_label2id    = None

        # convert everything to tensors ONCE here, not in __getitem__
        if input_ids is not None:
            self.input_ids      = input_ids if isinstance(input_ids, torch.Tensor) \
                                  else torch.tensor(input_ids,      dtype=torch.long)
            self.attention_masks = attention_masks if isinstance(attention_masks, torch.Tensor) \
                                  else torch.tensor(attention_masks, dtype=torch.long)
        else:
            self.input_ids       = None
            self.attention_masks = None

        # labels stay as lists until append_vocab() converts them,
        # so we defer tensor conversion to after append_vocab()
        self.d_labels  = d_labels
        self.labels    = labels
        self.word_masks = word_masks

    def _labels_to_tensors(self):
        """Call once after append_vocab() to lock everything into tensors."""
        self.labels    = torch.tensor(self.labels,    dtype=torch.long)
        self.d_labels  = torch.tensor(self.d_labels,  dtype=torch.long)
        self.word_masks = torch.tensor(self.word_masks, dtype=torch.long)

    def append_vocab(self, label2id, d_label2id):
        self.label2id   = label2id
        self.d_label2id = d_label2id
        if getattr(self, 'vocab_already_applied', False):
            print("  Skipping append_vocab — already applied in cache")
            return
        for i in range(len(self.labels)):
            self.labels[i]   = [label2id.get(l, label2id['<OOV>']) for l in self.labels[i]]
            self.d_labels[i] = [d_label2id[l] for l in self.d_labels[i]]
        # convert to tensors after vocab applied
        self.labels   = torch.tensor(self.labels,   dtype=torch.long)
        self.d_labels = torch.tensor(self.d_labels, dtype=torch.long)

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        if isinstance(self.input_ids, np.ndarray):
            return {
                'input_ids':      torch.from_numpy(self.input_ids[idx].copy()),
                'attention_mask': torch.from_numpy(self.attention_masks[idx].copy()),
                'd_labels':       self.d_labels[idx],
                'labels':         self.labels[idx],
                'word_masks':     torch.from_numpy(self.word_masks[idx].copy()),
            }
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'd_labels':       self.d_labels[idx],
            'labels':         self.labels[idx],
            'word_masks':     self.word_masks[idx],
        }

def align_labels_to_subwords(
    srcs,
    word_level_labels,
    tokenizer,
    cache_file,
    batch_size=10000,      # reduce from 50000 — less RAM per chunk
    max_length=128,
    keep_label='$KEEP',
    pad_token='<PAD>',
    correct_label='$CORRECT',
    incorrect_label='$INCORRECT',
    commit_fn=None,        # <-- callable to flush volume, passed from modal
    skip_tensor_mmaps=False,  # if True, don't write input_ids/attention_masks/word_masks (used when these already exist in cache and we're just rebuilding label mmaps
):
    import numpy as np
    import pickle

    n          = len(srcs)
    progress_file = cache_file + '.progress.pkl'  # tracks completed batches

    # ── Resume state ──────────────────────────────────────────────
    if os.path.exists(progress_file):
        with open(progress_file, 'rb') as f:
            progress = pickle.load(f)
        start_batch    = progress['last_completed_batch'] + 1
        subword_labels  = progress['subword_labels']
        subword_d_labels = progress['subword_d_labels']
        print(f"Resuming from batch {start_batch} ({start_batch * batch_size}/{n} sentences)")
    else:
        start_batch      = 0
        subword_labels   = []
        subword_d_labels = []
        print(f"Starting fresh, {n} sentences total")

    # ── Pre-allocate mmap (safe to call even if file exists) ──────
    mode = 'r+' if os.path.exists(cache_file + '.input_ids.mmap') else 'w+'
    input_ids_mm      = np.memmap(cache_file + '.input_ids.mmap',
                                   dtype='int64', mode=mode, shape=(n, max_length))
    attention_mask_mm = np.memmap(cache_file + '.attention_mask.mmap',
                                   dtype='int64', mode=mode, shape=(n, max_length))
    word_masks_mm     = np.memmap(cache_file + '.word_masks.mmap',
                                   dtype='int64', mode=mode, shape=(n, max_length))

    itr = list(range(0, n, batch_size))

    for batch_idx, i in enumerate(tqdm(itr)):
        # skip already completed batches
        if batch_idx < start_batch:
            continue

        batch_srcs   = srcs[i:i+batch_size]
        batch_labels = word_level_labels[i:i+batch_size]
        batch_n      = len(batch_srcs)

        encode = tokenizer(
            batch_srcs,
            max_length=max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            is_split_into_words=True
        )

        # in the batch loop, wrap mmap writes with the flag:
        if not skip_tensor_mmaps:
            input_ids_mm[i:i+batch_n]      = encode['input_ids'].numpy()
            attention_mask_mm[i:i+batch_n] = encode['attention_mask'].numpy()

        for j, wlabels in enumerate(batch_labels):
            d_labels_row = []
            labels_row   = []
            wmask        = []
            word_ids     = encode.word_ids(j)
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    labels_row.append(pad_token)
                    d_labels_row.append(pad_token)
                    wmask.append(0)
                elif word_idx != previous_word_idx:
                    l = wlabels[word_idx]
                    labels_row.append(l)
                    wmask.append(1)
                    d_labels_row.append(
                        incorrect_label if l != keep_label else correct_label
                    )
                else:
                    labels_row.append(pad_token)
                    d_labels_row.append(pad_token)
                    wmask.append(0)
                previous_word_idx = word_idx

            subword_labels.append(labels_row)
            subword_d_labels.append(d_labels_row)

            wmask_arr = np.array(wmask, dtype='int64')
            if len(wmask_arr) < max_length:
                wmask_arr = np.pad(wmask_arr, (0, max_length - len(wmask_arr)))
            word_masks_mm[i+j] = wmask_arr[:max_length]

        # ── Save progress every 5 batches ─────────────────────────
        if batch_idx % 5 == 0:
            input_ids_mm.flush()
            attention_mask_mm.flush()
            word_masks_mm.flush()

            with open(progress_file, 'wb') as f:
                pickle.dump({
                    'last_completed_batch': batch_idx,
                    'subword_labels':       subword_labels,
                    'subword_d_labels':     subword_d_labels,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            if commit_fn is not None:
                commit_fn()   # flush to Modal Volume

            print(f"  Progress saved at batch {batch_idx} ({i+batch_n}/{n} sentences)")

    # final flush
    input_ids_mm.flush()
    attention_mask_mm.flush()
    word_masks_mm.flush()

    # cleanup progress file — preprocessing complete
    if os.path.exists(progress_file):
        os.remove(progress_file)

    return subword_d_labels, subword_labels, \
           input_ids_mm, attention_mask_mm, word_masks_mm

def load_gector_format(
    input_file: str,
    delimeter: str='SEPL|||SEPR',
    additional_delimeter: str='SEPL__SEPR'
):  
    srcs = []
    word_level_labels = []  # the size will be (#sents, seq_length) if not get_interactive_tags,
                                # (#iteration, #sents, seq_length) if get_interactive_tags
    with open(input_file) as f:
        for line in f:
            src = [x.split(delimeter)[0] for x in line.split()]
            labels = [x.split(delimeter)[1] for x in line.split()]
            # Use only first tags. E.g. $REPLACE_meSEPL__SEPR$APPEND_too → $REPLACE_me
            labels = [l.split(additional_delimeter)[0] for l in labels]
            srcs.append(src)
            word_level_labels.append(labels)
    return srcs, word_level_labels
    
import pickle
import hashlib

CACHE_DIR = "/gector-data/cache"

def _cache_path(input_file: str, tokenizer, max_length: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)

    key = f"{input_file}_{tokenizer.name_or_path}_{max_length}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]

    filename = os.path.basename(input_file)

    return os.path.join(
        CACHE_DIR,
        f"{filename}.cache_{h}"
    )

def load_dataset(
    input_file: str,
    tokenizer: PreTrainedTokenizer,
    delimeter: str = 'SEPL|||SEPR',
    additional_delimeter: str = 'SEPL__SEPR',
    batch_size: int = 50000,
    max_length: int = 128,
    use_cache: bool = True,
    commit_fn=None,
):
    cache_file = _cache_path(input_file, tokenizer, max_length)
    meta_file  = cache_file + '.meta.pt'

    input_ids_path      = cache_file + '.input_ids.mmap'
    attention_mask_path = cache_file + '.attention_mask.mmap'
    word_masks_path     = cache_file + '.word_masks.mmap'
    labels_path         = cache_file + '.labels.mmap'
    d_labels_path       = cache_file + '.d_labels.mmap'

    # ── Helper: infer n from mmap file size ───────────────────────────────────
    def infer_n_from_mmap(path, max_length):
        return os.path.getsize(path) // (max_length * 4)  # int32 = 4 bytes

    # ── Helper: load minimal meta, rebuild if fat/corrupt ────────────────────
    def load_or_rebuild_meta():
        """
        Returns (n, vocab_applied).
        If meta is missing, corrupt, or fat (>1MB), infers n from mmap
        and writes a fresh minimal meta.
        """
        # no meta at all — infer from mmap if possible
        if not os.path.exists(meta_file):
            if os.path.exists(input_ids_path):
                n = infer_n_from_mmap(input_ids_path, max_length)
                has_label_mmaps = (
                    os.path.exists(labels_path) and
                    os.path.exists(d_labels_path)
                )
                _write_meta(meta_file, n, has_label_mmaps)
                print(f"  Rebuilt missing meta: n={n:,}, vocab_applied={has_label_mmaps}")
                return n, has_label_mmaps
            return None, False  # nothing exists yet

        # meta exists — check if it's fat (old format with srcs/labels lists)
        size_mb = os.path.getsize(meta_file) / 1e6
        if size_mb > 1.0:
            print(f"  Fat meta detected ({size_mb:.0f} MB) — rebuilding from mmap...")
            if os.path.exists(input_ids_path):
                n = infer_n_from_mmap(input_ids_path, max_length)
                has_label_mmaps = (
                    os.path.exists(labels_path) and
                    os.path.exists(d_labels_path)
                )
                _write_meta(meta_file, n, has_label_mmaps)
                print(f"  Rebuilt meta: n={n:,}, vocab_applied={has_label_mmaps}")
                return n, has_label_mmaps
            else:
                # fat meta but no mmaps — delete and reprocess from scratch
                os.remove(meta_file)
                return None, False

        # meta is small — try loading it
        try:
            meta = torch.load(meta_file, weights_only=True)
            return meta['n'], meta.get('vocab_applied', False)
        except Exception as e:
            print(f"  Corrupt meta ({e}) — rebuilding...")
            if os.path.exists(input_ids_path):
                n = infer_n_from_mmap(input_ids_path, max_length)
                has_label_mmaps = (
                    os.path.exists(labels_path) and
                    os.path.exists(d_labels_path)
                )
                _write_meta(meta_file, n, has_label_mmaps)
                return n, has_label_mmaps
            os.remove(meta_file)
            return None, False

    def _write_meta(path, n, vocab_applied):
        tmp = path + '.tmp'
        torch.save({'n': n, 'vocab_applied': vocab_applied}, tmp)
        os.replace(tmp, path)  # atomic on Linux

    # ── Check what exists ─────────────────────────────────────────────────────
    tensor_mmaps_exist = (
        os.path.exists(input_ids_path) and
        os.path.exists(attention_mask_path) and
        os.path.exists(word_masks_path)
    )

    if use_cache and tensor_mmaps_exist:
        n, vocab_applied = load_or_rebuild_meta()

        if n is not None:
            print(f"✓ Cache hit: {cache_file} (n={n:,})")

            input_ids_mm      = np.memmap(input_ids_path,
                                           dtype='int32', mode='r', shape=(n, max_length))
            attention_mask_mm = np.memmap(attention_mask_path,
                                           dtype='int32', mode='r', shape=(n, max_length))
            word_masks_mm     = np.memmap(word_masks_path,
                                           dtype='int32', mode='r', shape=(n, max_length))

            # ── Apply vocab to label mmaps if not done yet ────────────────
            if not vocab_applied:
                print("  Label mmaps missing — building from raw data + vocab...")
                # need to reload raw string labels from source file
                srcs, word_level_labels = load_gector_format(
                    input_file,
                    delimeter=delimeter,
                    additional_delimeter=additional_delimeter
                )
                assert len(srcs) == n, \
                    f"Source file row count ({len(srcs)}) != mmap n ({n})"

                # we need a tokenizer + vocab to build label mmaps
                # vocab must be passed in — caller should provide label2id
                # defer: return dataset with vocab_already_applied=False
                # so append_vocab() handles it the old way
                # (this path only hits if apply_vocab was never run)
                print("  WARNING: label mmaps not found and vocab not applied.")
                print("  Will apply vocab in-memory via append_vocab().")

                dataset = GECToRDataset(
                    srcs            = srcs,
                    d_labels        = None,   # built during align
                    labels          = None,
                    word_masks      = word_masks_mm,
                    input_ids       = input_ids_mm,
                    attention_masks = attention_mask_mm,
                    tokenizer       = tokenizer,
                    max_length      = max_length,
                )
                # re-run label alignment only (tensor mmaps already exist)
                d_labels, labels, _, _, _ = align_labels_to_subwords(
                    srcs,
                    word_level_labels,
                    tokenizer=tokenizer,
                    cache_file=cache_file,
                    batch_size=batch_size,
                    max_length=max_length,
                    commit_fn=commit_fn,
                    skip_tensor_mmaps=True,   # don't rewrite input_ids/mask/word_masks
                )
                dataset.d_labels = d_labels
                dataset.labels   = labels
                return dataset

            # ── Happy path: label mmaps exist ────────────────────────────
            labels_mm   = np.memmap(labels_path,
                                     dtype='int32', mode='r', shape=(n, max_length))
            d_labels_mm = np.memmap(d_labels_path,
                                     dtype='int32', mode='r', shape=(n, max_length))

            dataset = GECToRDataset(
                srcs            = None,
                d_labels        = d_labels_mm,
                labels          = labels_mm,
                word_masks      = word_masks_mm,
                input_ids       = input_ids_mm,
                attention_masks = attention_mask_mm,
                tokenizer       = tokenizer,
                max_length      = max_length,
            )
            dataset.vocab_already_applied = True
            return dataset

    # ── No cache — process from scratch ──────────────────────────────────────
    print(f"No cache found, processing {input_file} ...")
    srcs, word_level_labels = load_gector_format(
        input_file,
        delimeter=delimeter,
        additional_delimeter=additional_delimeter
    )

    d_labels, labels, input_ids_mm, attention_mask_mm, word_masks_mm = \
        align_labels_to_subwords(
            srcs,
            word_level_labels,
            tokenizer=tokenizer,
            cache_file=cache_file,
            batch_size=batch_size,
            max_length=max_length,
            commit_fn=commit_fn,
        )

    if use_cache:
        _write_meta(meta_file, len(srcs), vocab_applied=False)

    return GECToRDataset(
        srcs            = srcs,
        d_labels        = d_labels,
        labels          = labels,
        word_masks      = word_masks_mm,
        input_ids       = input_ids_mm,
        attention_masks = attention_mask_mm,
        tokenizer       = tokenizer,
        max_length      = max_length,
    )