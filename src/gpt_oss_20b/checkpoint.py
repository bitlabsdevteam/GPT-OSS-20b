from pathlib import Path
import torch


def save_checkpoint(path: str, model, optimizer, step: int) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, p)


def load_checkpoint(path: str, model, optimizer=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))
