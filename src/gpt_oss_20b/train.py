import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import load_config
from .dist import init_distributed, is_rank0
from .model import GPTModel, ModelConfig
from .parallel import infer_parallel_from_env, validate_parallel
from .profiler import timed_section, collect_step_stats
from .checkpoint import save_checkpoint
from .data import load_text, NextTokenDataset, sample_batch
from .tokenizer import CharTokenizer


def _autocast_dtype(enabled_bf16: bool):
    if enabled_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    cfg = load_config(args.config).raw
    init_distributed()

    par = infer_parallel_from_env()
    validate_parallel(par)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    train_cfg = cfg["train"]
    model_raw = cfg["model"]
    text_mode = bool(cfg.get("data", {}).get("train_path"))
    tokenizer = None
    dataset = None

    if text_mode:
        train_path = cfg["data"]["train_path"]
        text = load_text(train_path)
        tokenizer = CharTokenizer.train_from_text(text)
        dataset = NextTokenDataset(tokenizer.encode(text), block_size=int(model_raw["max_seq_len"]))
        vocab_size = tokenizer.vocab_size
    else:
        vocab_size = int(model_raw["vocab_size"])

    mcfg = ModelConfig(
        vocab_size=vocab_size,
        d_model=int(model_raw["d_model"]),
        n_heads=int(model_raw["n_heads"]),
        n_layers=int(model_raw["n_layers"]),
        ffn_mult=int(model_raw["ffn_mult"]),
        max_seq_len=int(model_raw.get("max_seq_len", 1024)),
        dropout=float(model_raw.get("dropout", 0.1)),
    )
    model = GPTModel(mcfg).to(device)
    if dist.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[local_rank] if device.startswith("cuda") else None)
    optim = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]))

    bsz = int(train_cfg["micro_batch_size"])
    seqlen = int(model_raw["max_seq_len"])
    max_steps = int(train_cfg["max_steps"])
    log_every = int(cfg["logging"]["log_every"])
    ckpt_every = int(cfg["logging"].get("ckpt_every", 200))
    use_bf16 = bool(train_cfg.get("bf16", True))
    checkpoint_dir = Path(cfg.get("logging", {}).get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if tokenizer is not None and is_rank0():
        tokenizer.save(checkpoint_dir / "tokenizer.json")

    for step in range(1, max_steps + 1):
        if dataset is not None:
            x, y = sample_batch(dataset, batch_size=bsz, device=device)
        else:
            x = torch.randint(0, vocab_size, (bsz, seqlen), device=device)
            y = torch.randint(0, vocab_size, (bsz, seqlen), device=device)

        with timed_section() as elapsed:
            optim.zero_grad(set_to_none=True)
            dtype = _autocast_dtype(use_bf16)
            if dtype is not None:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            loss.backward()
            optim.step()

        if step % log_every == 0 and is_rank0():
            stats = collect_step_stats(step, float(loss.item()), elapsed(), bsz * seqlen, device)
            print(
                f"step={stats.step} loss={stats.loss:.4f} dt={stats.dt_s:.3f}s "
                f"tok/s={stats.tokens_per_s:.1f} max_mem={stats.max_mem_gb:.2f}GB",
                flush=True,
            )

        if step % ckpt_every == 0 and is_rank0():
            module = model.module if hasattr(model, "module") else model
            save_checkpoint(str(checkpoint_dir / f"step_{step:07d}.pt"), module, optim, step, model_config=mcfg)

    if is_rank0():
        module = model.module if hasattr(model, "module") else model
        save_checkpoint(str(checkpoint_dir / "last.pt"), module, optim, max_steps, model_config=mcfg)


if __name__ == "__main__":
    main()
