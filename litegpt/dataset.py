import random
import os
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

PAD = 50256
BOS = 50257
EOS = 50258

class GPTDataset(Dataset):
    def __init__(self, ids, context_length=512, samples_per_epoch=1_000_000):
        self.ids = np.asarray(ids, dtype=np.uint16)
        self.context_length = context_length
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        start = random.randint(0, len(self.ids) - self.context_length - 1)
        length = random.randint(2, self.context_length + 1)

        seq = self.ids[start:start + length]
        x = np.concatenate([[BOS], seq])
        y = np.concatenate([seq, [EOS]])

        return torch.from_numpy(x).long(), torch.from_numpy(y).long()
    
def collate_fn(batch):
    x, y = zip(*batch)
    x = pad_sequence(x, batch_first=True, padding_value=PAD)
    y = pad_sequence(y, batch_first=True, padding_value=PAD)
    return x, y

def create_causal_mask(L, device):
    return torch.tril(torch.ones(L, L, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)

def create_pad_mask(x, pad_id=PAD):
    return (x != pad_id).unsqueeze(1).unsqueeze(2)

def create_attn_mask(x, pad_id=PAD):
    _, L = x.shape
    pad_mask , causal_mask = create_pad_mask(x, pad_id), create_causal_mask(L, device=x.device) # (B, 1, 1, L) and (1, 1, L, L)
    return pad_mask & causal_mask # (B, 1, L, L)


def load_tokenized_data(path: str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode='r')


# Preparation Functions

def get_datasets(path):
    train_ds = GPTDataset(load_tokenized_data(os.path.join(path, 'train.bin')))
    val_ds = GPTDataset(load_tokenized_data(os.path.join(path, 'val.bin')))
    return train_ds, val_ds



def tokenize_text(text: str, tokenizer, add_eos: bool = True) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if add_eos:
        ids.append(tokenizer.eos_id)
    return ids

def prepare_from_text_files(
    input_path:str,
    tokenizer,
    output_dir:str,
    max_tokens: int=10_000_000,
    val_ratio: float = 0.1
):
    if os.path.isfile(input_path):
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        tokens = tokenize_text(text, tokenizer, add_eos=False)
    elif os.path.isdir(input_path):
        tokens = []
        for fname in sorted(os.listdir(input_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(input_path, fname), 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                text = text.replace("\n\n", "\n")
                tokens.extend(tokenize_text(text, tokenizer, add_eos=False))
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")
    
    tokens = tokens[:max_tokens]
    
    n = int((1-val_ratio) * len(tokens))
    train_ids = np.array(tokens[:n], dtype=np.uint16)
    val_ids = np.array(tokens[n:], dtype=np.uint16)
    
    os.makedirs(output_dir, exist_ok=True)
    train_ids.tofile(os.path.join(output_dir, 'train.bin'))
    val_ids.tofile(os.path.join(output_dir, 'val.bin'))
    
    print(f"Train: {len(train_ids):,} tokens")
    print(f"Val:   {len(val_ids):,} tokens")
    print(f"Saved to {output_dir}")
    
def prepare_from_hf_dataset(
    dataset_name: str,
    tokenizer,
    output_dir: str,
    max_tokens: int | None = None,
    text_field: str = "text",
    train_split: str = "train",
    val_split: str = "validation",
    streaming: bool = True,
):
    """Tokenize HuggingFace dataset → train.bin, val.bin (streaming)"""
    from datasets import load_dataset

    train_ds = load_dataset(dataset_name, split=train_split, streaming=streaming)
    val_ds = load_dataset(dataset_name, split=val_split, streaming=streaming)

    # Tokenize train
    train_tokens = []
    for sample in train_ds:
        train_tokens.extend(tokenize_text(sample[text_field], tokenizer, add_eos=True))
        if max_tokens is not None: 
            if len(train_tokens) >= max_tokens:
                break
    if max_tokens is not None:
        train_tokens = train_tokens[:max_tokens]

    # Tokenize val (full or ~10% of train)
    val_limit = max_tokens // 10 if max_tokens is not None else None
    val_tokens = []
    for sample in val_ds:
        val_tokens.extend(tokenize_text(sample[text_field], tokenizer, add_eos=True))
        if val_limit is not None:
            if len(val_tokens) >= val_limit:
                break

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.array(train_tokens, dtype=np.uint16).tofile(os.path.join(output_dir, 'train.bin'))
    np.array(val_tokens, dtype=np.uint16).tofile(os.path.join(output_dir, 'val.bin'))

    print(f"Train: {len(train_tokens):,} tokens")
    print(f"Val:   {len(val_tokens):,} tokens")
    print(f"Saved to {output_dir}")