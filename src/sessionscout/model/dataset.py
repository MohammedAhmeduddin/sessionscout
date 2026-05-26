"""
model/dataset.py — PyTorch Dataset for session sequences.

What this file does:
  Wraps sequences.parquet into a PyTorch Dataset so the
  training loop can load batches efficiently.

  Three things it produces per session:
    input_ids      — the 64 token IDs as a LongTensor
    attention_mask — 1 where real tokens are, 0 where PAD is
    label          — 1 if purchased, 0 if not (FloatTensor)

  Why the attention mask matters:
    Without it, the Transformer treats PAD tokens (0s) as real
    events and learns from noise. The mask tells the model
    "ignore these positions — they are empty."

  Example:
    sequence:        [0, 0, 0, 1, 2, 4, 1]  (3 PADs + 4 real)
    attention_mask:  [0, 0, 0, 1, 1, 1, 1]
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Optional

from sessionscout.config import cfg


class SessionDataset(Dataset):
    """
    PyTorch Dataset wrapping sequences.parquet.

    Args:
        df: DataFrame with columns: sequence, label
            (output of build_sequence_dataset())
    """

    def __init__(self, df: pd.DataFrame):
        # Stack all sequences into a 2D numpy array: (N, 64)
        # then convert to tensor once — much faster than per-item conversion
        sequences = np.stack(df["sequence"].values)  # (N, 64)

        self.input_ids = torch.tensor(sequences, dtype=torch.long)
        # Attention mask: 1 where token is NOT padding, 0 where it IS padding
        self.attention_mask = (self.input_ids != cfg.vocab.pad).float()
        self.labels = torch.tensor(df["label"].values, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.input_ids[idx],       # shape: (64,)
            "attention_mask": self.attention_mask[idx],   # shape: (64,)
            "label":          self.labels[idx],           # scalar
        }


def load_datasets(
    sequences_path: Optional[Path] = None,
    val_size: float = None,
    test_size: float = None,
    random_seed: int = None,
) -> Tuple[SessionDataset, SessionDataset, SessionDataset]:
    """
    Load sequences.parquet and split into train / val / test datasets.

    Returns three SessionDataset objects: train, val, test.

    The split is stratified on the label so each split has
    the same conversion rate as the full dataset.
    """
    path = sequences_path or cfg.paths.sequences_parquet
    val_size = val_size or cfg.sequence.val_size
    test_size = test_size or cfg.sequence.test_size
    seed = random_seed or cfg.sequence.random_seed

    if not path.exists():
        raise FileNotFoundError(
            f"sequences.parquet not found at {path}\n"
            "Run first: python -m sessionscout.features.sequences"
        )

    df = pd.read_parquet(path)

    # First split off the test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    # Then split train into train + val
    # Adjust val_size to account for the smaller pool
    adjusted_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    train_ds = SessionDataset(train_df.reset_index(drop=True))
    val_ds   = SessionDataset(val_df.reset_index(drop=True))
    test_ds  = SessionDataset(test_df.reset_index(drop=True))

    return train_ds, val_ds, test_ds


def make_dataloaders(
    train_ds: SessionDataset,
    val_ds: SessionDataset,
    test_ds: SessionDataset,
    batch_size: int = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap datasets in DataLoaders for training.

    Train loader shuffles. Val and test loaders do not.
    """
    bs = batch_size or cfg.training.batch_size

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False, num_workers=0
    )

    return train_loader, val_loader, test_loader
