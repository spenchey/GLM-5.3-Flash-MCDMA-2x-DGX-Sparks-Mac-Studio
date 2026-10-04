from __future__ import annotations

import pytest

from experiments.three_machine.cuda_bf16_qk import patch_cuda_source


def test_patch_rounds_all_serial_cuda_qk_writes_to_bfloat16() -> None:
    source = """
            x[lane * 4 + i] = y;
            dst[base + lane * 4 + i] = y;
            dst[base + lane * 4 + i] = y;
    """

    patched = patch_cuda_source(source)

    assert "x[lane * 4 + i] = bf(y);" in patched
    assert patched.count("dst[base + lane * 4 + i] = bf(y);") == 2
    assert " = y;" not in patched


@pytest.mark.parametrize(
    "source",
    (
        "x[lane * 4 + i] = y;\ndst[base + lane * 4 + i] = y;",
        "x[lane * 4 + i] = bf(y);\n"
        "dst[base + lane * 4 + i] = bf(y);\n"
        "dst[base + lane * 4 + i] = bf(y);",
    ),
)
def test_patch_refuses_incomplete_or_already_modified_source(source: str) -> None:
    with pytest.raises(RuntimeError, match="unexpected TensorFold KDA source"):
        patch_cuda_source(source)
