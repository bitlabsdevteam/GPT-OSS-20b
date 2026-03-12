from pathlib import Path
import torch
from gpt_oss_20b.model import ModelConfig, GPTModel
from gpt_oss_20b.checkpoint import save_checkpoint, load_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path):
    cfg = ModelConfig(vocab_size=100, d_model=32, n_heads=4, n_layers=1, ffn_mult=2)
    m1 = GPTModel(cfg)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(str(path), m1, opt1, step=7, model_config=cfg)

    m2 = GPTModel(cfg)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    step = load_checkpoint(str(path), m2, opt2)
    assert step == 7

    payload = load_checkpoint(str(path), map_location="cpu")
    assert payload["step"] == 7
    assert payload["model_config"]["vocab_size"] == 100
