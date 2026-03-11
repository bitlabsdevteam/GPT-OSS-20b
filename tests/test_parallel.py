import pytest
from gpt_oss_20b.parallel import ParallelConfig, validate_parallel


def test_parallel_validate_ok():
    validate_parallel(ParallelConfig(dp=1, tp=1, pp=1, ep=1, sp=False))


def test_parallel_validate_sp_requires_tp():
    with pytest.raises(ValueError):
        validate_parallel(ParallelConfig(dp=1, tp=1, pp=1, ep=1, sp=True))
