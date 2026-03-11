import time
from contextlib import contextmanager
from dataclasses import dataclass
import torch


@dataclass
class StepStats:
    step: int
    loss: float
    dt_s: float
    tokens_per_s: float
    max_mem_gb: float


@contextmanager
def timed_section():
    t0 = time.perf_counter()
    yield lambda: time.perf_counter() - t0


def collect_step_stats(step: int, loss: float, dt_s: float, batch_tokens: int, device: str) -> StepStats:
    tps = float(batch_tokens) / max(dt_s, 1e-8)
    mem = 0.0
    if device.startswith("cuda") and torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / (1024**3)
    return StepStats(step=step, loss=loss, dt_s=dt_s, tokens_per_s=tps, max_mem_gb=mem)
