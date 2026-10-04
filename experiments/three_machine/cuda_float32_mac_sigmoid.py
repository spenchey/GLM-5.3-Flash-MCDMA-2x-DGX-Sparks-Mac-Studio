"""Test exact projections with Metal-style bfloat16 sigmoid boundaries."""

from __future__ import annotations

from typing import Any

from experiments.three_machine.cuda_bf16_qk import (
    install_kda_variant,
    patch_cuda_source as patch_qk_source,
)
from experiments.three_machine.cuda_float32_qmm import _float32_projection
from experiments.three_machine.cuda_scalar_qmm import _scalar_projection
from experiments.three_machine.cuda_torch_qmm import install_torch_projection_variant


VARIANT = "float32-qmm-layer0-scalar-aux-mac-bf16-sigmoid"
EXPERIMENT_EXTENSION = "tensorfold_glm_kda_v2_macbfsig_exactqmm"
_SIGMOID_HELPER = (
    "__device__ __forceinline__ float sigmoidf_(float x) { "
    "return 1.0f / (1.0f + expf(-x)); }"
)
_BF16_SIGMOID_HELPER = _SIGMOID_HELPER + r"""

__device__ __forceinline__ float sigmoidbf_(float x) {
    const float xb = bf(x);
    const float e = bf(expf(fabsf(xb)));
    const float y = bf(1.0f / bf(1.0f + e));
    return xb < 0.0f ? y : bf(1.0f - y);
}"""
_ACTIVATION = "const float act = bf(acc / (1.0f + expf(-acc)));"
_BF16_ACTIVATION = (
    "const float xb = bf(acc);\n"
    "            const float act = bf(xb * sigmoidbf_(xb));"
)
_CHAIN_BETA = (
    "beta_s = bf(sigmoidf_(__bfloat162float("
    "P[(size_t)r * p_stride + b_off + h])));"
)
_BF16_CHAIN_BETA = (
    "beta_s = sigmoidbf_(__bfloat162float("
    "P[(size_t)r * p_stride + b_off + h]));"
)
_WIDE_BETA = (
    "b_save[r * H + h] = bf(sigmoidf_(__bfloat162float("
    "P[(size_t)r * p_stride + b_off + h])));"
)
_BF16_WIDE_BETA = (
    "b_save[r * H + h] = sigmoidbf_(__bfloat162float("
    "P[(size_t)r * p_stride + b_off + h]));"
)


def patch_cuda_source(source: str) -> str:
    """Match Metal's stable bfloat16 sigmoid steps, then q/k stores."""

    expected = {
        _SIGMOID_HELPER: 1,
        _ACTIVATION: 3,
        _CHAIN_BETA: 1,
        _WIDE_BETA: 2,
    }
    for text, count in expected.items():
        if source.count(text) != count:
            raise RuntimeError(
                "unexpected TensorFold KDA sigmoid source: "
                f"expected {count} occurrence(s)"
            )
    patched = source.replace(_SIGMOID_HELPER, _BF16_SIGMOID_HELPER)
    patched = patched.replace(_ACTIVATION, _BF16_ACTIVATION)
    patched = patched.replace(_CHAIN_BETA, _BF16_CHAIN_BETA)
    patched = patched.replace(_WIDE_BETA, _BF16_WIDE_BETA)
    return patch_qk_source(patched)


def install_combined_variant() -> dict[str, Any]:
    """Install sigmoid/KDA source parity before exact layer-0 projections."""

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
