from pathlib import Path
import torch


def save_checkpoint(path: str, model, optimizer, step: int, model_config=None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
    }
    if model_config is not None:
        payload["model_config"] = model_config.__dict__
    torch.save(payload, p)


def load_checkpoint(path: str, model=None, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        return int(ckpt.get("step", 0))
    if model is not None:
        return int(ckpt.get("step", 0))
    return ckpt
