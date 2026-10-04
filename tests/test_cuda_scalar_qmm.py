from __future__ import annotations

from pathlib import Path

from experiments.three_machine import cuda_scalar_qmm


ROOT = Path(__file__).resolve().parents[1]


def test_scalar_diagnostic_uses_unique_extension_and_ordered_float32_fma() -> None:
    python_source = (ROOT / "experiments/three_machine/cuda_scalar_qmm.py").read_text()
    cpp_source = (ROOT / "experiments/three_machine/scalar_qmm.cpp").read_text()
    cuda_source = (ROOT / "experiments/three_machine/scalar_qmm.cu").read_text()

    assert cuda_scalar_qmm.VARIANT == "scalar-qmm-layer0"
    assert 'name="glm_mcdma_scalar_qmm_v1"' in python_source
    assert "__fmaf_rn" in cuda_source
    assert "#pragma unroll 1" in cuda_source
    assert "mma" not in cuda_source.lower()
    assert "k % gs == 0" in cpp_source
    assert "k % (8 * gs)" not in cpp_source
