"""Diagnose layer-0 projection parity with a plain ordered CUDA reduction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from experiments.three_machine.cuda_torch_qmm import (
    install_torch_projection_variant,
)


VARIANT = "scalar-qmm-layer0"
_UNPACKED: dict[int, tuple[Any, Any, Any]] = {}


@lru_cache(maxsize=1)
def _extension():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(
        name="glm_mcdma_scalar_qmm_v1",
        sources=[str(here / "scalar_qmm.cpp"), str(here / "scalar_qmm.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def _scalar_projection(x: Any, quantized: Any, out: Any) -> None:
    from tensorfold.cuda.kernels import qmm

    key = id(quantized)
    unpacked = _UNPACKED.get(key)
    if unpacked is None:
        unpacked = qmm.unpack(quantized)
        _UNPACKED[key] = unpacked
    words, scales, biases = unpacked
    _extension().scalar_qmm(
        x,
        words,
        scales,
        biases,
        out,
        int(quantized.gs),
    )


def main() -> int:
    install_torch_projection_variant(
        projector=_scalar_projection,
        variant=VARIANT,
    )
    from experiments.three_machine import cuda_prefill

    return cuda_prefill.main()


if __name__ == "__main__":
    raise SystemExit(main())
