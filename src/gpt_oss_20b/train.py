import argparse
import time
import torch
import torch.nn.functional as F

from .config import load_config
from .dist import init_distributed, is_rank0
from .model import TinyGPT


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    cfg = load_config(args.config).raw
    init_distributed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyGPT(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))

    bsz = int(cfg["train"]["micro_batch_size"])
    seqlen = int(cfg["model"]["max_seq_len"])
    vocab = int(cfg["model"]["vocab_size"])

    for step in range(1, int(cfg["train"]["max_steps"]) + 1):
        x = torch.randint(0, vocab, (bsz, seqlen), device=device)
        y = torch.randint(0, vocab, (bsz, seqlen), device=device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab), y.view(-1))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step % int(cfg["logging"]["log_every"]) == 0 and is_rank0():
            print(f"step={step} loss={loss.item():.4f} t={time.time():.0f}", flush=True)


if __name__ == "__main__":
    main()
