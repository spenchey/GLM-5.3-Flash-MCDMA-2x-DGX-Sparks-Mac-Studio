"""Run Spark prefill with the Mac prompt KDA's explicit roundings.

The Mac prompt kernel rounds the convolution accumulation to bfloat16 before
SiLU, rounds the SiLU result again, and stores normalized q/k as bfloat16.  The
pinned CUDA kernel otherwise performs the activation on unrounded float32 and
keeps normalized q/k in float32.  This disposable-container experiment changes
only those six source sites and compiles them under a distinct extension name.
"""

from __future__ import annotations

from typing import Any

from experiments.three_machine.cuda_bf16_qk import (
    install_kda_variant,
    patch_cuda_source as patch_qk_source,
)


EXPERIMENT_EXTENSION = "tensorfold_glm_kda_v2_macprefill"
_ACTIVATION = "const float act = bf(acc / (1.0f + expf(-acc)));"
_MAC_ACTIVATION = (
    "const float xb = bf(acc);\n"
    "            const float act = bf(xb / (1.0f + expf(-xb)));"
)


def patch_cuda_source(source: str) -> str:
    """Apply the Mac prompt KDA's conv-activation and q/k boundaries."""

    if source.count(_ACTIVATION) != 3:
        raise RuntimeError(
            "unexpected TensorFold KDA activation source: expected three "
            "convolution activation sites"
        )
    return patch_qk_source(source.replace(_ACTIVATION, _MAC_ACTIVATION))


def install_mac_kda_variant() -> dict[str, Any]:
    """Install the complete Mac-prompt-rounding diagnostic variant."""

    return install_kda_variant(
        patch_cuda_source,
        extension=EXPERIMENT_EXTENSION,
        variant="mac-kda-rounding",
    )


def main() -> int:
    install_mac_kda_variant()
    from experiments.three_machine import cuda_prefill

    return cuda_prefill.main()


if __name__ == "__main__":
    raise SystemExit(main())
