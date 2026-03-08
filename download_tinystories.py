"""
Download and tokenize the TinyStories dataset.

Usage:
    pip install datasets tiktoken numpy
    python download_tinystories.py

Outputs:
    data/tinystories/train.bin   – uint16 numpy array of token ids
    data/tinystories/val.bin     – uint16 numpy array of token ids
    data/tinystories/meta.pkl    – vocab size metadata

Target: first ~100 M tokens (configurable via MAX_TOKENS).
Train/val split: 99% / 1%.

Tokens are encoded with the GPT-2 BPE tokenizer (tiktoken cl100k_base gives
the same vocab size; we use gpt2 to stay consistent with nanoGPT defaults).
"""

import os
import pickle
import struct

import numpy as np

MAX_TOKENS = 100_000_000   # 100 M tokens total
VAL_FRACTION = 0.01        # 1 % held out for validation
OUT_DIR = os.path.join('data', 'tinystories')

DATASET_NAME = 'roneneldan/TinyStories'
SPLIT = 'train'            # TinyStories only has a train split on HF


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install the 'datasets' package:  pip install datasets")

    try:
        import tiktoken
    except ImportError:
        raise SystemExit("Install the 'tiktoken' package:  pip install tiktoken")

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {DATASET_NAME} …")
    ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True, trust_remote_code=True)

    enc = tiktoken.get_encoding('gpt2')
    vocab_size = enc.n_vocab  # 50257

    all_tokens: list[int] = []
    n_docs = 0

    print(f"Tokenising up to {MAX_TOKENS:,} tokens …")
    for example in ds:
        text: str = example.get('text', '') or ''
        tokens = enc.encode_ordinary(text)
        tokens.append(enc.eot_token)  # <|endoftext|> between documents
        all_tokens.extend(tokens)
        n_docs += 1
        if len(all_tokens) >= MAX_TOKENS:
            break
        if n_docs % 10_000 == 0:
            print(f"  {n_docs:,} docs, {len(all_tokens):,} tokens …")

    all_tokens = all_tokens[:MAX_TOKENS]
    print(f"Total tokens collected: {len(all_tokens):,}")

    arr = np.array(all_tokens, dtype=np.uint16)

    n_val = max(1, int(len(arr) * VAL_FRACTION))
    n_train = len(arr) - n_val

    train_arr = arr[:n_train]
    val_arr = arr[n_train:]

    train_path = os.path.join(OUT_DIR, 'train.bin')
    val_path = os.path.join(OUT_DIR, 'val.bin')
    meta_path = os.path.join(OUT_DIR, 'meta.pkl')

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    with open(meta_path, 'wb') as f:
        pickle.dump({'vocab_size': vocab_size}, f)

    print(f"Wrote {n_train:,} train tokens  → {train_path}")
    print(f"Wrote {n_val:,}  val tokens    → {val_path}")
    print(f"Wrote metadata (vocab_size={vocab_size}) → {meta_path}")


if __name__ == '__main__':
    main()
