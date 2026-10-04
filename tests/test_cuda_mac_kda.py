from __future__ import annotations

import pytest

from experiments.three_machine.cuda_mac_kda import patch_cuda_source


def _source() -> str:
    return """
            const float act = bf(acc / (1.0f + expf(-acc)));
            const float act = bf(acc / (1.0f + expf(-acc)));
            const float act = bf(acc / (1.0f + expf(-acc)));
            x[lane * 4 + i] = y;
            dst[base + lane * 4 + i] = y;
            dst[base + lane * 4 + i] = y;
    """


def test_patch_matches_mac_convolution_and_qk_rounding() -> None:
    patched = patch_cuda_source(_source())

    assert patched.count("const float xb = bf(acc);") == 3
    assert patched.count("bf(xb / (1.0f + expf(-xb)))") == 3
    assert "bf(acc / (1.0f + expf(-acc)))" not in patched
    assert "x[lane * 4 + i] = bf(y);" in patched
    assert patched.count("dst[base + lane * 4 + i] = bf(y);") == 2


@pytest.mark.parametrize(
    "source",
    (
        _source().replace(
            "const float act = bf(acc / (1.0f + expf(-acc)));\n", "", 1
        ),
        _source().replace(
            "const float act = bf(acc / (1.0f + expf(-acc)));",
            "const float xb = bf(acc);\n"
            "            const float act = bf(xb / (1.0f + expf(-xb)));",
        ),
    ),
)
def test_patch_refuses_incomplete_or_already_modified_activation(source: str) -> None:
    with pytest.raises(RuntimeError, match="unexpected TensorFold KDA activation source"):
        patch_cuda_source(source)
