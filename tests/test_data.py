import torch
from gpt_oss_20b.data import SyntheticTokenDataGenerator


def test_synthetic_generator_is_deterministic():
    g1 = SyntheticTokenDataGenerator(vocab_size=100, seq_len=8, batch_size=2, seed=123)
    g2 = SyntheticTokenDataGenerator(vocab_size=100, seq_len=8, batch_size=2, seed=123)

    x1, y1 = g1.next_batch(device="cpu")
    x2, y2 = g2.next_batch(device="cpu")

    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)
