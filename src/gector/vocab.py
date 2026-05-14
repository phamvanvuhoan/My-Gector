from .configuration import GECToRConfig
from .dataset import GECToRDataset
import os
import torch
from tqdm import tqdm


def build_vocab(
    train_dataset: GECToRDataset,
    n_max_labels: int=5000,
    n_max_d_labels: int=2
):
    label2id = {'<OOV>':0, '$KEEP':1}
    d_label2id = {'$CORRECT':0, '$INCORRECT':1, '<PAD>':2}
    freq_labels, _ = train_dataset.get_labels_freq(
        exluded_labels=['<PAD>'] + list(label2id.keys())
    )

    def get_high_freq(freq: dict, n_max: int):
        descending_freq = sorted(
            freq.items(), key=lambda x:x[1], reverse=True
        )
        high_freq = [x[0] for x in descending_freq][:n_max]
        if len(high_freq) < n_max:
            print(f'Warning: the size of the vocablary: {len(high_freq)} is less than n_max: {n_max}.')
        return high_freq
    
    high_freq_labels = get_high_freq(freq_labels, n_max_labels-2)
    for i, x in enumerate(high_freq_labels):
        label2id[x] = i + 2
    label2id['<PAD>'] = len(label2id)
    return label2id, d_label2id

def load_vocab_from_config(config_file: str):
    config = GECToRConfig.from_pretrained(config_file, not_dir=True)
    return config.label2id, config.d_label2id

def load_vocab_from_official(dir):
    vocab_path = os.path.join(dir, 'labels.txt')
    vocab = open(vocab_path).read().replace('@@PADDING@@', '').replace('@@UNKNOWN@@', '').rstrip().split('\n')
    label2id = {'<OOV>':0, '$KEEP':1}
    d_label2id = {'$CORRECT':0, '$INCORRECT':1, '<PAD>':2}
    idx = len(label2id)
    for v in vocab:
        if v not in label2id:
            label2id[v] = idx
            idx += 1
    label2id['<PAD>'] = idx
    return label2id, d_label2id

def compute_class_weights(
    dataset: GECToRDataset,
    label2id: dict,
    strategy: str = 'sqrt_inverse_freq',
    max_weight: float = 10.0,
) -> torch.Tensor:
    """
    Count label frequencies from the mmap and return per-class weights.

    The mmap is int32; we read in chunks and accumulate into a float64
    counter — no cast to int64 needed.
    """
    n_labels = len(label2id)
    counts   = torch.zeros(n_labels, dtype=torch.float64)
    pad_id   = label2id['<PAD>']

    print("Computing class weights from label mmap ...")
    chunk = 100_000
    for i in tqdm(range(0, len(dataset), chunk)):
        # dataset.labels is int32 mmap; read a slice and cast to int64 for
        # scatter_add_ (which requires a Long index tensor).
        batch = torch.from_numpy(
            dataset.labels[i : i + chunk].copy()
        ).to(torch.long).flatten()
        valid = batch[batch != pad_id]
        counts.scatter_add_(0, valid, torch.ones_like(valid, dtype=torch.float64))

    counts = counts.float().clamp(min=1.0)
    if strategy == 'inverse_freq':
        weights = 1.0 / counts
    elif strategy == 'sqrt_inverse_freq':
        weights = 1.0 / counts.sqrt()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    weights = weights / weights.mean()
    weights = weights.clamp(max=max_weight)
    weights[pad_id] = 0.0

    # Drop the <PAD> slot — the projection layer has (num_labels - 1) outputs.
    keep_ids = [i for i in range(n_labels) if i != pad_id]
    return weights[keep_ids]