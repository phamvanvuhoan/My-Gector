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
        # convert to tensors immediately after vocab is applied
        self._labels_to_tensors()

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        # ZERO tensor construction — pure tensor indexing only
        if self.input_ids is not None:
            return {
                'input_ids':      self.input_ids[idx],
                'attention_mask': self.attention_masks[idx],
                'd_labels':       self.d_labels[idx],
                'labels':         self.labels[idx],
                'word_masks':     self.word_masks[idx],
            }
        # fallback if no cached input_ids (backward compat)
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
    batch_size: int=100000,
    max_length: int=128,
    keep_label: str='$KEEP',
    pad_token: str='<PAD>',
    correct_label: str='$CORRECT',
    incorrect_label: str='$INCORRECT'
):
    itr = list(range(0, len(srcs), batch_size))
    subword_labels = []
    subword_d_labels = []
    word_masks = []
    all_input_ids = []
    all_attention_masks = []
    for i in tqdm(itr):
        encode = tokenizer(
            srcs[i:i+batch_size],
            max_length=max_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            is_split_into_words=True
        )
        # store as tensors, not numpy
        all_input_ids.append(encode['input_ids'])
        all_attention_masks.append(encode['attention_mask'])

        for i, wlabels in enumerate(word_labels[i:i+batch_size]):
            d_labels = []
            labels = []
            wmask = []
            word_ids = encode.word_ids(i)
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
                    if l != keep_label:
                        d_labels.append(incorrect_label)
                    else:
                        d_labels.append(correct_label)
                else:
                    labels.append(pad_token)
                    d_labels.append(pad_token)
                    wmask.append(0)
                previous_word_idx = word_idx
            subword_d_labels.append(d_labels)
            subword_labels.append(labels)
            word_masks.append(wmask)
    
    # cat all batches into single tensors
    input_ids_tensor      = torch.cat(all_input_ids,      dim=0)
    attention_masks_tensor = torch.cat(all_attention_masks, dim=0)

    return subword_d_labels, subword_labels, word_masks, input_ids_tensor, attention_masks_tensor

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

def _cache_path(input_file: str, tokenizer, max_length: int) -> str:
    """Generate a unique cache filename based on file + tokenizer + config."""
    # hash the tokenizer name and max_length so changing them invalidates cache
    key = f"{input_file}_{tokenizer.name_or_path}_{max_length}"
    h = hashlib.md5(key.encode()).hexdigest()[:8]
    return input_file + f".cache_{h}.pkl"

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

    if use_cache and os.path.exists(cache_file):
        print(f"Loading cached dataset from {cache_file} ...")
        data = torch.load(cache_file)
        return GECToRDataset(
            srcs            = data['srcs'],
            d_labels        = data['d_labels'],
            labels          = data['labels'],
            word_masks      = data['word_masks'],
            input_ids       = data['input_ids'],
            attention_masks = data['attention_masks'],
            tokenizer       = tokenizer,
            max_length      = max_length,
        )

    # cache miss — do the full preprocessing
    srcs, word_level_labels = load_gector_format(
        input_file,
        delimeter=delimeter,
        additional_delimeter=additional_delimeter
    )
    d_labels, labels, word_masks, input_ids, attention_masks = align_labels_to_subwords(
        srcs,
        word_level_labels,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )

    if use_cache:
        print(f"Saving cache to {cache_file} ...")
        torch.save({               # torch.save handles tensors better than pickle
            'srcs':            srcs,
            'd_labels':        d_labels,
            'labels':          labels,
            'word_masks':      word_masks,
            'input_ids':       input_ids,        # already a tensor
            'attention_masks': attention_masks,   # already a tensor
        }, cache_file)

    return GECToRDataset(
        srcs            = srcs,
        d_labels        = d_labels,
        labels          = labels,
        word_masks      = word_masks,
        input_ids       = input_ids,
        attention_masks = attention_masks,
        tokenizer       = tokenizer,
        max_length      = max_length,
    )