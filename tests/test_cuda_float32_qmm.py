from __future__ import annotations

from pathlib import Path

from experiments.three_machine import cuda_float32_qmm


ROOT = Path(__file__).resolve().parents[1]


def test_float32_diagnostic_disables_tf32_and_has_a_distinct_variant() -> None:
    source = (ROOT / "experiments/three_machine/cuda_float32_qmm.py").read_text()

    assert cuda_float32_qmm.VARIANT == "float32-qmm-layer0-scalar-aux-serial-kda"
    assert "allow_tf32 = False" in source
    assert 'set_float32_matmul_precision("highest")' in source
    assert "x.to(torch.float32)" in source
    assert "auxiliary_projector=_scalar_projection" in source
