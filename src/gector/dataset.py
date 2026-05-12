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
        for i in range(len(self.labels)):
            self.labels[i]   = [label2id.get(l, label2id['<OOV>']) for l in self.labels[i]]
            self.d_labels[i] = [d_label2id[l] for l in self.d_labels[i]]
        # convert to tensors after vocab applied
        self.labels   = torch.tensor(self.labels,   dtype=torch.long)
        self.d_labels = torch.tensor(self.d_labels, dtype=torch.long)

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        # input_ids and attention_mask — handle both mmap and tensor
        if isinstance(self.input_ids, np.ndarray):
            # mmap path
            input_ids      = torch.from_numpy(np.array(self.input_ids[idx],      dtype='int32')).long()
            attention_mask = torch.from_numpy(np.array(self.attention_masks[idx], dtype='int32')).long()
            word_masks     = torch.from_numpy(np.array(self.word_masks[idx],      dtype='int32')).long()
        else:
            # tensor path (fallback)
            input_ids      = self.input_ids[idx]
            attention_mask = self.attention_masks[idx]
            word_masks     = self.word_masks[idx]

        # labels are always tensors after append_vocab()
        return {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'd_labels':       self.d_labels[idx],
            'labels':         self.labels[idx],
            'word_masks':     word_masks,
        }

def align_labels_to_subwords(
    srcs: List[str],
    word_level_labels: List[List[str]],
    tokenizer: PreTrainedTokenizer,
    cache_file,
    batch_size: int=50000,
    max_length: int=128,
    keep_label: str='$KEEP',
    pad_token: str='<PAD>',
    correct_label: str='$CORRECT',
    incorrect_label: str='$INCORRECT'
):
    n = len(srcs)
    print(f"Total sentences: {n}")

    # pre-allocate memory-mapped arrays on disk — never fully in RAM
    input_ids_mm      = np.memmap(cache_file + '.input_ids.mmap',
                                   dtype='int32', mode='w+', shape=(n, max_length))
    attention_mask_mm = np.memmap(cache_file + '.attention_mask.mmap',
                                   dtype='int32', mode='w+', shape=(n, max_length))
    word_masks_mm     = np.memmap(cache_file + '.word_masks.mmap',
                                   dtype='int32', mode='w+', shape=(n, max_length))
    # labels stay as lists until vocab is applied, then converted later
    subword_labels   = []
    subword_d_labels = []

    itr = list(range(0, n, batch_size))
    written = 0

    for i in tqdm(itr):
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

        # write directly to mmap — no accumulation in RAM
        input_ids_mm[written:written+batch_n]      = encode['input_ids'].numpy()
        attention_mask_mm[written:written+batch_n] = encode['attention_mask'].numpy()

        for j, wlabels in enumerate(batch_labels):
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
                    d_labels.append(
                        incorrect_label if l != keep_label else correct_label
                    )
                else:
                    labels.append(pad_token)
                    d_labels.append(pad_token)
                    wmask.append(0)
                previous_word_idx = word_idx
            subword_labels.append(labels)
            subword_d_labels.append(d_labels)
            wmask_arr = np.array(wmask, dtype='int32')
            # pad or truncate wmask to max_length
            if len(wmask_arr) < max_length:
                wmask_arr = np.pad(wmask_arr, (0, max_length - len(wmask_arr)))
            word_masks_mm[written+j] = wmask_arr[:max_length]

        written += batch_n

        # flush periodically to avoid OS page cache buildup
        if (i // batch_size) % 10 == 0:
            input_ids_mm.flush()
            attention_mask_mm.flush()
            word_masks_mm.flush()

    input_ids_mm.flush()
    attention_mask_mm.flush()
    word_masks_mm.flush()

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
    use_cache: bool = True,        # <-- add
):
    cache_file = _cache_path(input_file, tokenizer, max_length)
    meta_file  = cache_file + '.meta.pt'

    if use_cache and os.path.exists(meta_file):
        print(f"✓ Cache hit: {cache_file}")
        meta = torch.load(meta_file)
        n    = meta['n']

        # load as mmap — only pages actually accessed are read from disk
        input_ids_mm      = np.memmap(cache_file + '.input_ids.mmap',
                                       dtype='int32', mode='r', shape=(n, max_length))
        attention_mask_mm = np.memmap(cache_file + '.attention_mask.mmap',
                                       dtype='int32', mode='r', shape=(n, max_length))
        word_masks_mm     = np.memmap(cache_file + '.word_masks.mmap',
                                       dtype='int32', mode='r', shape=(n, max_length))
        return GECToRDataset(
            srcs            = meta['srcs'],
            d_labels        = meta['d_labels'],
            labels          = meta['labels'],
            word_masks      = word_masks_mm,
            input_ids       = input_ids_mm,
            attention_masks = attention_mask_mm,
            tokenizer       = tokenizer,
            max_length      = max_length,
        )

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
            max_length=max_length
        )

    if use_cache:
        print(f"Saving cache metadata to {meta_file} ...")
        torch.save({
            'n':        len(srcs),
            'srcs':     srcs,
            'd_labels': d_labels,   # still lists, vocab not applied yet
            'labels':   labels,
        }, meta_file)

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