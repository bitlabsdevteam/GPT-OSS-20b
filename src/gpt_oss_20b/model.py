import torch
import torch.nn as nn


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int = 50304, d_model: int = 1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        h = self.ln(h)
        return self.head(h)
