from __future__ import annotations

import pytest

from experiments.three_machine.cuda_float32_mac_sigmoid import patch_cuda_source


def _source() -> str:
    return """
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
            const float act = bf(acc / (1.0f + expf(-acc)));
            const float act = bf(acc / (1.0f + expf(-acc)));
            const float act = bf(acc / (1.0f + expf(-acc)));
            beta_s = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
        if (i == 0) b_save[r * H + h] = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
        if (i == 0) b_save[r * H + h] = bf(sigmoidf_(__bfloat162float(P[(size_t)r * p_stride + b_off + h])));
            x[lane * 4 + i] = y;
            dst[base + lane * 4 + i] = y;
            dst[base + lane * 4 + i] = y;
    """


def test_patch_matches_bfloat16_sigmoid_steps_and_qk_stores() -> None:
    patched = patch_cuda_source(_source())

    assert patched.count("sigmoidbf_(float x)") == 1
    assert patched.count("const float act = bf(xb * sigmoidbf_(xb));") == 3
    assert patched.count("beta_s = sigmoidbf_") == 1
    assert patched.count("b_save[r * H + h] = sigmoidbf_") == 2
    assert patched.count("x[lane * 4 + i] = bf(y);") == 1
    assert patched.count("dst[base + lane * 4 + i] = bf(y);") == 2
    assert "bf(sigmoidf_" not in patched


@pytest.mark.parametrize(
    "source",
    (
        _source().replace(
            "const float act = bf(acc / (1.0f + expf(-acc)));\n", "", 1
        ),
        _source().replace("beta_s = bf(sigmoidf_", "beta_s = sigmoidbf_"),
    ),
)
def test_patch_refuses_incomplete_or_already_modified_source(source: str) -> None:
    with pytest.raises(RuntimeError, match="unexpected TensorFold KDA sigmoid source"):
        patch_cuda_source(source)
