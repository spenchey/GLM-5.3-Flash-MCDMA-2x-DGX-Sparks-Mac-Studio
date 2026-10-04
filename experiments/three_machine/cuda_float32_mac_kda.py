"""Combine exact layer-0 projections with Mac prompt KDA boundaries.

This diagnostic first installs the pinned CUDA source patch that matches the
Mac prompt kernel's explicit bfloat16 storage boundaries.  It then replaces
only layer 0's three quantized projections with the exact reference paths that
already matched the Mac byte-for-byte.  The order matters: TensorFold's KDA
module must be patched before the forward module imports it.
"""

from __future__ import annotations

from typing import Any

from experiments.three_machine.cuda_bf16_qk import install_kda_variant
from experiments.three_machine.cuda_float32_qmm import _float32_projection
from experiments.three_machine.cuda_mac_kda import patch_cuda_source
from experiments.three_machine.cuda_scalar_qmm import _scalar_projection
from experiments.three_machine.cuda_torch_qmm import install_torch_projection_variant


VARIANT = "float32-qmm-layer0-scalar-aux-mac-kda"
EXPERIMENT_EXTENSION = "tensorfold_glm_kda_v2_macprefill_exactqmm"


def install_combined_variant() -> dict[str, Any]:
    """Install KDA source parity first, then exact layer-0 projections."""

    kda = install_kda_variant(
        patch_cuda_source,
        extension=EXPERIMENT_EXTENSION,
        variant=VARIANT,
    )
    projection = install_torch_projection_variant(
        projector=_float32_projection,
        auxiliary_projector=_scalar_projection,
        variant=VARIANT,
    )
    return {"variant": VARIANT, "kda": kda, "projection": projection}


def main() -> int:
    install_combined_variant()
    from experiments.three_machine import cuda_prefill

    return cuda_prefill.main()


if __name__ == "__main__":
    raise SystemExit(main())
