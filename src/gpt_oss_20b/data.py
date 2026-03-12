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


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def sample_batch(dataset: NextTokenDataset, batch_size: int, device: str):
    starts = [random.randint(0, len(dataset) - 1) for _ in range(batch_size)]
    xs, ys = zip(*(dataset[idx] for idx in starts))
    x = torch.stack(xs).to(device)
    y = torch.stack(ys).to(device)
    return x, y
