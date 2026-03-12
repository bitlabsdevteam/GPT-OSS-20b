from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}

    @classmethod
    def train_from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: idx for idx, ch in enumerate(chars)}
        return cls(stoi)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unknown = [ch for ch in text if ch not in self.stoi]
        if unknown:
            missing = ''.join(sorted(set(unknown)))
            raise ValueError(f"Prompt contains unseen characters: {missing!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return ''.join(self.itos[idx] for idx in token_ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"stoi": self.stoi}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({str(k): int(v) for k, v in payload["stoi"].items()})
