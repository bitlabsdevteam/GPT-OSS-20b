from gpt_oss_20b.config import load_config


def test_load_config():
    cfg = load_config("configs/train_3xa100.yaml").raw
    assert "model" in cfg
    assert cfg["model"]["vocab_size"] > 0
