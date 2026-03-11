from dataclasses import dataclass
import os


@dataclass
class ParallelConfig:
    dp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    sp: bool = False

    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp


def infer_parallel_from_env() -> ParallelConfig:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    # defensible default for 3x A100: TP=1, PP=1, DP=world
    return ParallelConfig(dp=world, tp=1, pp=1, ep=1, sp=False)


def validate_parallel(cfg: ParallelConfig) -> None:
    if cfg.dp < 1 or cfg.tp < 1 or cfg.pp < 1 or cfg.ep < 1:
        raise ValueError("All parallel degrees must be >= 1")
    if cfg.sp and cfg.tp == 1:
        raise ValueError("Sequence parallel requires tensor parallel > 1")
