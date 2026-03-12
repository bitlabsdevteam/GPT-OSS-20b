import torch
from gpt_oss_20b.model import ModelConfig, GPTModel


def test_model_forward_shape():
    cfg = ModelConfig(vocab_size=1000, d_model=64, n_heads=8, n_layers=2, ffn_mult=2)
    model = GPTModel(cfg)
    x = torch.randint(0, 1000, (2, 16))
    y = model(x)
    assert y.shape == (2, 16, 1000)


def test_model_generate_grows_sequence():
    cfg = ModelConfig(vocab_size=128, d_model=32, n_heads=4, n_layers=1, ffn_mult=2, max_seq_len=32)
    model = GPTModel(cfg)
    x = torch.randint(0, 128, (1, 8))
    y = model.generate(x, max_new_tokens=5, temperature=1.0, top_k=10)
    assert y.shape == (1, 13)
