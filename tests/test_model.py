import torch
from gpt_oss_20b.model import ModelConfig, GPTModel


def test_model_forward_shape():
    cfg = ModelConfig(vocab_size=1000, d_model=64, n_heads=8, n_layers=2, ffn_mult=2)
    model = GPTModel(cfg)
    x = torch.randint(0, 1000, (2, 16))
    y = model(x)
    assert y.shape == (2, 16, 1000)
