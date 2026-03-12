import argparse
import torch

from .checkpoint import load_checkpoint
from .model import GPTModel, ModelConfig
from .tokenizer import CharTokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--tokenizer", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = load_checkpoint(args.checkpoint, map_location=device)
    model_cfg = ModelConfig(**payload["model_config"])
    model = GPTModel(model_cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    tokenizer = CharTokenizer.load(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    y = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
