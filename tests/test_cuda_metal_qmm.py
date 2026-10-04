from __future__ import annotations

import pytest

from experiments.three_machine.cuda_metal_qmm import (
    _BF16_FMA2,
    _FLOAT32_FMA2,
    patch_cuda_source,
)


def test_patch_matches_metal_q4_dequantization_order() -> None:
    source = f"before\n{_BF16_FMA2}\nafter\n"

    patched = patch_cuda_source(source)

    assert "fma.rn.bf16x2" not in patched
    assert patched.count("__bfloat1622float2") == 3
    assert patched.count("fmaf(") == 2
    assert "__floats2bfloat162_rn" in patched
    assert _FLOAT32_FMA2 in patched
    assert patched.startswith("before\n")
    assert patched.endswith("\nafter\n")


@pytest.mark.parametrize(
    "source",
    (
        "unrelated source",
        _FLOAT32_FMA2,
        f"{_BF16_FMA2}\n{_BF16_FMA2}",
    ),
)
def test_patch_refuses_missing_already_modified_or_duplicate_source(source: str) -> None:
    with pytest.raises(RuntimeError, match="unexpected TensorFold QMM source"):
        patch_cuda_source(source)
