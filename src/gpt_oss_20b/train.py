import argparse
import os
import torch
import torch.nn.functional as F

from .config import load_config
from .dist import init_distributed, is_rank0
from .model import GPTModel, ModelConfig
from .parallel import infer_parallel_from_env, validate_parallel
from .profiler import timed_section, collect_step_stats


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

    mcfg = ModelConfig(
        vocab_size=int(cfg["model"]["vocab_size"]),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        ffn_mult=int(cfg["model"]["ffn_mult"]),
    )
    model = GPTModel(mcfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))

    bsz = int(cfg["train"]["micro_batch_size"])
    seqlen = int(cfg["model"]["max_seq_len"])
    vocab = int(cfg["model"]["vocab_size"])
    max_steps = int(cfg["train"]["max_steps"])
    log_every = int(cfg["logging"]["log_every"])
    use_bf16 = bool(cfg["train"].get("bf16", True))

    for step in range(1, max_steps + 1):
        x = torch.randint(0, vocab, (bsz, seqlen), device=device)
        y = torch.randint(0, vocab, (bsz, seqlen), device=device)

        with timed_section() as elapsed:
            optim.zero_grad(set_to_none=True)
            dtype = _autocast_dtype(use_bf16)
            if dtype is not None:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))

            loss.backward()
            optim.step()

        if step % log_every == 0 and is_rank0():
            stats = collect_step_stats(step, float(loss.item()), elapsed(), bsz * seqlen, device)
            print(
                f"step={stats.step} loss={stats.loss:.4f} dt={stats.dt_s:.3f}s "
                f"tok/s={stats.tokens_per_s:.1f} max_mem={stats.max_mem_gb:.2f}GB",
                flush=True,
            )


if __name__ == "__main__":
    main()
