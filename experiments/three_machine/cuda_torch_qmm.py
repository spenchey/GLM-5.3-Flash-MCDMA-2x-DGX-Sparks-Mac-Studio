"""Diagnose layer-0 projection parity with CUDA's standard BF16 matmul.

Only the first KDA layer's wide input projection changes.  Its MLX Q4 weights
are dequantized once to bfloat16, then multiplied with PyTorch/cuBLAS using
its native bfloat16 matrix path.  Every other projection and layer remains on
the pinned TensorFold kernels.  This is a disposable diagnostic, not a
production path.
"""

from __future__ import annotations

import importlib
import json
from types import ModuleType
from typing import Any, Callable


VARIANT = "torch-qmm-layer0"
TARGET_LAYER = 0
_DEQUANTIZED_BFLOAT16: dict[int, Any] = {}


def dequantized_bfloat16_weight(quantized: Any) -> Any:
    import torch
    from tensorfold.cuda.kernels import qmm

    key = id(quantized)
    weight = _DEQUANTIZED_BFLOAT16.get(key)
    if weight is None:
        words, scales, biases = qmm.unpack(quantized)
        shifts = torch.arange(8, device=words.device, dtype=torch.int32) * 4
        values = ((words[:, :, None] >> shifts) & 0xF).reshape(
            quantized.n,
            quantized.k // quantized.gs,
            quantized.gs,
        )
        weight = (
            values.to(torch.float32)
            * scales.to(torch.float32)[:, :, None]
            + biases.to(torch.float32)[:, :, None]
        ).to(torch.bfloat16).reshape(quantized.n, quantized.k).contiguous()
        _DEQUANTIZED_BFLOAT16[key] = weight
    return weight


def _torch_projection(x: Any, quantized: Any, out: Any) -> None:
    import torch

    weight = dequantized_bfloat16_weight(quantized)

    # Both Metal and the TensorFold prompt kernel retain float32 accumulators.
    # Disable CUDA's optional reduced-precision reduction for this diagnostic.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    out.copy_(torch.mm(x, weight.t()))


def install_torch_projection_variant(
    *,
    forward_module: ModuleType | Any | None = None,
    projector: Callable[[Any, Any, Any], None] | None = None,
    auxiliary_projector: Callable[[Any, Any, Any], None] | None = None,
    variant: str = VARIANT,
    emit: bool = True,
) -> dict[str, Any]:
    """Replace only layer 0's KDA front projection and fail on double install."""

    forward = forward_module or importlib.import_module(
        "tensorfold.families.glm5_next.cuda.forward"
    )
    if hasattr(forward, "_glm_mcdma_original_kda_front"):
        raise RuntimeError("the torch-qmm layer-0 diagnostic is already installed")
    original = forward.kda_front
    project = projector or _torch_projection

    def torch_kda_front(layer, buffers, lo: int, hi: int) -> None:
        if int(layer.index) != TARGET_LAYER:
            return original(layer, buffers, lo, hi)
        kda = layer.kda
        projection = buffers.kproj[0, lo:hi]
        project(buffers.normed[lo:hi], kda.proj, projection)
        if auxiliary_projector is None:
            forward.mm(
                buffers,
                projection[:, kda.fa_off:kda.fa_off + 128],
                kda.fb,
                None,
                buffers.ka[lo:hi],
            )
            forward.mm(
                buffers,
                projection[:, kda.ga_off:kda.ga_off + 128],
                kda.gb,
                None,
                buffers.kg[lo:hi],
            )
        else:
            auxiliary_projector(
                projection[:, kda.fa_off:kda.fa_off + 128],
                kda.fb,
                buffers.ka[lo:hi],
            )
            auxiliary_projector(
                projection[:, kda.ga_off:kda.ga_off + 128],
                kda.gb,
                buffers.kg[lo:hi],
            )

    forward._glm_mcdma_original_kda_front = original
    forward.kda_front = torch_kda_front
    event = {
        "event": "glm_cuda_projection_variant",
        "layer": TARGET_LAYER,
        "variant": variant,
    }
    if emit:
        print(json.dumps(event, sort_keys=True), flush=True)
    return event


def main() -> int:
    install_torch_projection_variant()
    from experiments.three_machine import cuda_prefill

    return cuda_prefill.main()


if __name__ == "__main__":
    raise SystemExit(main())
