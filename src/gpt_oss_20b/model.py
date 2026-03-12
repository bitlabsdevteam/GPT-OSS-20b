from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    ffn_mult: int
    max_seq_len: int = 1024
    dropout: float = 0.1


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * ffn_mult)
        self.fc2 = nn.Linear(d_model * ffn_mult, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + self.dropout(h)
        h = self.ln2(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return x + self.dropout(h)


class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_mult, cfg.dropout) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.cfg.max_seq_len}")

        pos = torch.arange(seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        h = self.embed(x) + self.pos_embed(pos)
        attn_mask = self._causal_mask(seq_len, x.device)
        for blk in self.blocks:
            h = blk(h, attn_mask)
        h = self.ln_f(h)
        return self.head(h)

    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            x_cond = x[:, -self.cfg.max_seq_len :]
            logits = self(x_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)
        return x
