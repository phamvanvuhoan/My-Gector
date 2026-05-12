from typing import List, Tuple
from collections import Counter
import torch
from tqdm import tqdm
import os
from transformers import PreTrainedTokenizer

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
        self.tokenizer      = tokenizer
        self.srcs           = srcs
        self.max_length     = max_length
        self.label2id       = None
        self.d_label2id     = None

        if input_ids is not None:
            self.input_ids       = input_ids if isinstance(input_ids, torch.Tensor) \
                                   else torch.tensor(input_ids, dtype=torch.long)
            self.attention_masks = attention_masks if isinstance(attention_masks, torch.Tensor) \
                                   else torch.tensor(attention_masks, dtype=torch.long)
        else:
            self.input_ids       = None
            self.attention_masks = None

        self.d_labels   = d_labels
        self.labels     = labels
        self.word_masks = word_masks

    def _labels_to_tensors(self):
        self.labels     = torch.tensor(self.labels,     dtype=torch.long)
        self.d_labels   = torch.tensor(self.d_labels,   dtype=torch.long)
        self.word_masks = torch.tensor(self.word_masks, dtype=torch.long)

    def append_vocab(self, label2id, d_label2id):
        self.label2id   = label2id
        self.d_label2id = d_label2id
        for i in range(len(self.labels)):
            self.labels[i]   = [label2id.get(l, label2id['<OOV>']) for l in self.labels[i]]
            self.d_labels[i] = [d_label2id[l] for l in self.d_labels[i]]
        self._labels_to_tensors()

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        if self.input_ids is not None:
            return {
                'input_ids':      self.input_ids[idx],
                'attention_mask': self.attention_masks[idx],
                'd_labels':       self.d_labels[idx],
                'labels':         self.labels[idx],
                'word_masks':     self.word_masks[idx],
            }
        src    = self.srcs[idx]
        encode = self.tokenizer(
            src,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=True
        )
        return {
            'input_ids':      encode['input_ids'].squeeze(),
            'attention_mask': encode['attention_mask'].squeeze(),
            'd_labels':       self.d_labels[idx],
            'labels':         self.labels[idx],
            'word_masks':     self.word_masks[idx],
        }


def align_labels_to_subwords(
    srcs: List[str],
    word_labels: List[List[str]],
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 100000,
    max_length: int = 128,
    keep_label: str = '$KEEP',
    pad_token: str = '<PAD>',
    correct_label: str = '$CORRECT',
    incorrect_label: str = '$INCORRECT'
):
    itr = list(range(0, len(srcs), batch_size))
    subword_labels   = []
    subword_d_labels = []
    word_masks       = []
    all_input_ids    = []
    all_attention_masks = []

    for i in tqdm(itr):
        encode = tokenizer(
            srcs[i:i + batch_size],
            max_length=max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            is_split_into_words=True
        )
        all_input_ids.append(encode['input_ids'])
        all_attention_masks.append(encode['attention_mask'])

        for j, wlabels in enumerate(word_labels[i:i + batch_size]):
            d_labels = []
            labels   = []
            wmask    = []
            word_ids = encode.word_ids(j)
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    labels.append(pad_token)
                    d_labels.append(pad_token)
                    wmask.append(0)
                elif word_idx != previous_word_idx:
                    l = wlabels[word_idx]
                    labels.append(l)
                    wmask.append(1)
                    d_labels.append(incorrect_label if l != keep_label else correct_label)
                else:
                    labels.append(pad_token)
                    d_labels.append(pad_token)
                    wmask.append(0)
                previous_word_idx = word_idx
            subword_d_labels.append(d_labels)
            subword_labels.append(labels)
            word_masks.append(wmask)

    input_ids_tensor       = torch.cat(all_input_ids,       dim=0)
    attention_masks_tensor = torch.cat(all_attention_masks, dim=0)

    return subword_d_labels, subword_labels, word_masks, input_ids_tensor, attention_masks_tensor


def load_gector_format(
    input_file: str,
    delimeter: str = 'SEPL|||SEPR',
    additional_delimeter: str = 'SEPL__SEPR'
):
    srcs             = []
    word_level_labels = []
    with open(input_file) as f:
        for line in f:
            src    = [x.split(delimeter)[0] for x in line.split()]
            labels = [x.split(delimeter)[1] for x in line.split()]
            labels = [l.split(additional_delimeter)[0] for l in labels]
            srcs.append(src)
            word_level_labels.append(labels)
    return srcs, word_level_labels


import hashlib

# ── Sharded cache helpers ──────────────────────────────────────────────────────
#
# Stage 1 has ~8.8 M sentences. Holding all tokenised tensors in RAM at once
# before torch.save causes a silent OOM kill at around 26% progress.
# The fix: process SHARD_SIZE sentences at a time, save each shard immediately,
# then mmap-load them back as a ConcatDataset — peak RAM stays bounded.

SHARD_SIZE = 500_000   # sentences per shard; tune down to 200k if still OOM


def _cache_key(input_file: str, tokenizer, max_length: int) -> str:
    key = f"{input_file}_{tokenizer.name_or_path}_{max_length}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _shard_path(input_file: str, cache_key: str, shard_idx: int) -> str:
    return f"{input_file}.cache_{cache_key}_shard{shard_idx:04d}.pt"


def _manifest_path(input_file: str, cache_key: str) -> str:
    return f"{input_file}.cache_{cache_key}_manifest.pt"


def _save_shard(
    path: str,
    srcs,
    d_labels,
    labels,
    word_masks,
    input_ids: torch.Tensor,
    attention_masks: torch.Tensor,
):
    torch.save({
        'srcs':            srcs,
        'n':               len(srcs),   # cached length — __init__ reads this, never reopens tensors
        'd_labels':        d_labels,
        'labels':          labels,
        'word_masks':      word_masks,
        'input_ids':       input_ids,
        'attention_masks': attention_masks,
    }, path)


class ShardedGECToRDataset(torch.utils.data.Dataset):
    """
    A dataset that reads shards lazily — only one shard lives in RAM at a time.

    This avoids the second OOM that occurred when _load_shards tried to
    torch.cat all 18 × ~3 Gi shards simultaneously.

    Layout per shard file (as saved by _save_shard):
        {srcs, d_labels, labels, word_masks, input_ids, attention_masks}

    After append_vocab() is called, label strings are converted to ids and
    stored back into each shard file so subsequent epochs skip re-conversion.
    """

    def __init__(self, shard_paths: list, shard_lengths: list, tokenizer, max_length: int):
        self.shard_paths  = shard_paths
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        self.label2id     = None
        self.d_label2id   = None

        # Lengths come from the manifest — we never open a shard file here.
        # Opening all 18 shards just to count rows was the OOM that crashed at shard 8.
        self._shard_sizes = shard_lengths
        self._cum_sizes   = [0]
        for n in shard_lengths:
            self._cum_sizes.append(self._cum_sizes[-1] + n)

        # Cache for the currently loaded shard (only one lives in RAM at a time)
        self._cached_shard_idx  = -1
        self._cached_shard_data = None

    # ── vocab ──────────────────────────────────────────────────────────────────

    def append_vocab(self, label2id: dict, d_label2id: dict):
        """
        Convert string labels → integer ids in every shard and re-save.
        Called once from train.py before the DataLoader is created.
        """
        self.label2id   = label2id
        self.d_label2id = d_label2id

        for shard_path in self.shard_paths:
            print(f"  [vocab] converting {shard_path} …")
            data = torch.load(shard_path, weights_only=False)

            # Skip if already converted (int tensor, not list of strings)
            if isinstance(data['labels'], torch.Tensor):
                continue

            labels_int   = [[label2id.get(l, label2id['<OOV>']) for l in row]
                            for row in data['labels']]
            d_labels_int = [[d_label2id[l] for l in row]
                            for row in data['d_labels']]

            data['labels']    = torch.tensor(labels_int,   dtype=torch.long)
            data['d_labels']  = torch.tensor(d_labels_int, dtype=torch.long)
            data['word_masks'] = torch.tensor(data['word_masks'], dtype=torch.long)

            torch.save(data, shard_path)

        # Invalidate cache so next __getitem__ reloads from updated files
        self._cached_shard_idx  = -1
        self._cached_shard_data = None

    # ── indexing ───────────────────────────────────────────────────────────────

    def __len__(self):
        return self._cum_sizes[-1]

    def _load_shard(self, shard_idx: int):
        if self._cached_shard_idx != shard_idx:
            self._cached_shard_data = torch.load(
                self.shard_paths[shard_idx], weights_only=False
            )
            self._cached_shard_idx = shard_idx

    def _find_shard(self, global_idx: int):
        """Binary search → (shard_idx, local_idx)."""
        lo, hi = 0, len(self._shard_sizes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if global_idx < self._cum_sizes[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        return lo, global_idx - self._cum_sizes[lo]

    def __getitem__(self, global_idx: int):
        shard_idx, local_idx = self._find_shard(global_idx)
        self._load_shard(shard_idx)
        d = self._cached_shard_data
        return {
            'input_ids':      d['input_ids'][local_idx],
            'attention_mask': d['attention_masks'][local_idx],
            'd_labels':       d['d_labels'][local_idx],
            'labels':         d['labels'][local_idx],
            'word_masks':     d['word_masks'][local_idx],
        }


def _load_shards(manifest: dict, tokenizer, max_length: int) -> ShardedGECToRDataset:
    """Return a lazy ShardedGECToRDataset — no tensors are cat'd in RAM."""
    print(f"  Building lazy ShardedGECToRDataset over "
          f"{len(manifest['shard_paths'])} shards …")
    return ShardedGECToRDataset(
        shard_paths   = manifest['shard_paths'],
        shard_lengths = manifest['shard_lengths'],   # read from manifest, never open shards
        tokenizer     = tokenizer,
        max_length    = max_length,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def preprocess_dataset(
    input_file: str,
    tokenizer: PreTrainedTokenizer,
    delimeter: str = 'SEPL|||SEPR',
    additional_delimeter: str = 'SEPL__SEPR',
    batch_size: int = 50000,
    max_length: int = 128,
    shard_size: int = SHARD_SIZE,
) -> None:
    """
    Tokenise and cache a dataset to shards. Does NOT load anything into RAM.
    Call this from the preprocessing job.
    """
    load_dataset(
        input_file           = input_file,
        tokenizer            = tokenizer,
        delimeter            = delimeter,
        additional_delimeter = additional_delimeter,
        batch_size           = batch_size,
        max_length           = max_length,
        use_cache            = True,
        shard_size           = shard_size,
        load_after_cache     = False,   # preprocess only — skip the RAM load
    )


def load_dataset(
    input_file: str,
    tokenizer: PreTrainedTokenizer,
    delimeter: str = 'SEPL|||SEPR',
    additional_delimeter: str = 'SEPL__SEPR',
    batch_size: int = 50000,
    max_length: int = 128,
    use_cache: bool = True,
    shard_size: int = SHARD_SIZE,
    load_after_cache: bool = True,   # set False in preprocess to skip RAM load
) -> 'ShardedGECToRDataset | None':
    """
    Tokenise, cache, and (optionally) load a dataset.

    load_after_cache=True  (default): tokenise → save shards → return ShardedGECToRDataset
    load_after_cache=False (preprocess only): tokenise → save shards → return None
    """
    cache_key      = _cache_key(input_file, tokenizer, max_length)
    manifest_file  = _manifest_path(input_file, cache_key)

    # ── Cache hit: manifest exists ───────────────────────────────────────────
    if use_cache and os.path.exists(manifest_file):
        print(f"Found manifest {manifest_file} …")
        manifest = torch.load(manifest_file, weights_only=False)

        if 'shard_lengths' not in manifest:
            print("  Manifest missing shard_lengths (old format) — rebuilding.")
        else:
            missing = [p for p in manifest['shard_paths'] if not os.path.exists(p)]
            if not missing:
                # All shards present — fast path
                print(f"  All {len(manifest['shard_paths'])} shards present.")
                if not load_after_cache:
                    return None
                return _load_shards(manifest, tokenizer, max_length)
            else:
                # Some shards missing — parse the source file but skip
                # existing shards; only recompute the missing ones.
                print(f"  {len(missing)} shard(s) missing — will recompute only those.")
                # Fall through to the shard loop below with _existing_manifest
                _existing_manifest = manifest

    # ── Partial or full recompute ─────────────────────────────────────────────
    # _existing_manifest is set when the manifest existed but some shards were
    # missing. In that case we parse the source file but skip existing shards.
    _existing_manifest = locals().get('_existing_manifest', None)

    print(f"Parsing {input_file} …")
    all_srcs, all_word_labels = load_gector_format(
        input_file,
        delimeter=delimeter,
        additional_delimeter=additional_delimeter
    )
    n_total = len(all_srcs)
    print(f"  {n_total:,} sentences loaded.")

    shard_paths   = []
    shard_lengths = []   # built during loop — no re-opening shards later
    n_shards      = (n_total + shard_size - 1) // shard_size

    # Build a fast lookup of which shard indices are already cached
    # — either from the existing manifest or from the file system
    if _existing_manifest:
        existing_shard_paths = set(_existing_manifest['shard_paths'])
    else:
        existing_shard_paths = set()

    for shard_idx in range(n_shards):
        shard_file = _shard_path(input_file, cache_key, shard_idx)

        # Skip already-saved shards so a mid-run crash is resumable,
        # AND skip shards that are confirmed present in an existing manifest
        shard_file_str = str(shard_file)
        if use_cache and (os.path.exists(shard_file) or shard_file_str in existing_shard_paths):
            if not os.path.exists(shard_file):
                # Listed in manifest but file actually gone — must recompute
                print(f"  Shard {shard_idx+1}/{n_shards} listed in manifest but missing on disk — recomputing.")
            else:
                print(f"  Shard {shard_idx+1}/{n_shards} already cached — skipping.")
                shard_paths.append(shard_file)
                shard_lengths.append(min(shard_size, n_total - shard_idx * shard_size))
                continue

        lo = shard_idx * shard_size
        hi = min(lo + shard_size, n_total)
        print(f"\n  Shard {shard_idx+1}/{n_shards}: sentences {lo:,}–{hi:,}")

        srcs_chunk   = all_srcs[lo:hi]
        labels_chunk = all_word_labels[lo:hi]

        d_labels, labels, word_masks, input_ids, attention_masks = \
            align_labels_to_subwords(
                srcs_chunk,
                labels_chunk,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_length=max_length,
            )

        if use_cache:
            print(f"  Saving shard → {shard_file}")
            _save_shard(
                shard_file,
                srcs_chunk, d_labels, labels, word_masks,
                input_ids, attention_masks
            )
            # Free tensors immediately after saving
            del d_labels, labels, word_masks, input_ids, attention_masks
            import gc; gc.collect()

        shard_paths.append(shard_file)
        shard_lengths.append(hi - lo)

    # Write manifest — lengths were collected during the loop, no re-opening needed
    if use_cache:
        torch.save(
            {'shard_paths': shard_paths, 'shard_lengths': shard_lengths},
            manifest_file
        )
        print(f"\n✓ All {n_shards} shards cached. Manifest → {manifest_file}")

    manifest = {'shard_paths': shard_paths, 'shard_lengths': shard_lengths}
    if not load_after_cache:
        return None   # preprocess path — don't load tensors into RAM
    return _load_shards(manifest, tokenizer, max_length)