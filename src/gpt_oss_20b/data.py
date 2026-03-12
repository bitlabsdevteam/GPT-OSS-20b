from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: list[int], block_size: int):
        if len(token_ids) <= block_size:
            raise ValueError(f"Need more than block_size={block_size} tokens, got {len(token_ids)}")
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx: int):
        chunk = self.token_ids[idx : idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class SyntheticTokenDataGenerator:
    """Deterministic synthetic token batches for perf and smoke testing."""

    def __init__(self, vocab_size: int, seq_len: int, batch_size: int, seed: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def next_batch(self, device: str):
        x = torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.seq_len),
            generator=self.generator,
            dtype=torch.long,
        )
        y = torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.seq_len),
            generator=self.generator,
            dtype=torch.long,
        )
        return x.to(device), y.to(device)


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def sample_batch(dataset: NextTokenDataset, batch_size: int, device: str):
    starts = [random.randint(0, len(dataset) - 1) for _ in range(batch_size)]
    xs, ys = zip(*(dataset[idx] for idx in starts))
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    return x, y
