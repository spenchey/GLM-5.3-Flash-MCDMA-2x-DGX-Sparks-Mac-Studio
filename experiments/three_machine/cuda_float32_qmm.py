"""Diagnose layer-0 parity with a fast full-float32 CUDA matrix multiply."""

from __future__ import annotations

from typing import Any

from experiments.three_machine.cuda_torch_qmm import (
    dequantized_bfloat16_weight,
    install_torch_projection_variant,
)
from experiments.three_machine.cuda_scalar_qmm import _scalar_projection


VARIANT = "float32-qmm-layer0-scalar-aux-serial-kda"
_FLOAT32: dict[int, Any] = {}


def _float32_projection(x: Any, quantized: Any, out: Any) -> None:
    import torch

    key = id(quantized)
    weight = _FLOAT32.get(key)
    if weight is None:
        weight = dequantized_bfloat16_weight(quantized).to(torch.float32)
        _FLOAT32[key] = weight

    # BF16 inputs have exact FP32 representations. Disable TF32 so cuBLAS uses
    # full FP32 products and accumulators before the single BF16 output cast.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    out.copy_(torch.mm(x.to(torch.float32), weight.t()))


def main() -> int:
    install_torch_projection_variant(
        projector=_float32_projection,
        auxiliary_projector=_scalar_projection,
        variant=VARIANT,
    )
    from experiments.three_machine import cuda_prefill

    return cuda_prefill.main()


if __name__ == "__main__":
    raise SystemExit(main())
