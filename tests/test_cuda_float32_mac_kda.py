from __future__ import annotations

from experiments.three_machine import cuda_float32_mac_kda


def test_combined_variant_installs_kda_before_projection(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def install_kda(patcher, *, extension: str, variant: str):
        calls.append(("kda", (patcher, extension, variant)))
        return {"event": "kda"}

    def install_projection(*, projector, auxiliary_projector, variant: str):
        calls.append(
            ("projection", (projector, auxiliary_projector, variant))
        )
        return {"event": "projection"}

    monkeypatch.setattr(cuda_float32_mac_kda, "install_kda_variant", install_kda)
    monkeypatch.setattr(
        cuda_float32_mac_kda,
        "install_torch_projection_variant",
        install_projection,
    )

    event = cuda_float32_mac_kda.install_combined_variant()

    assert [name for name, _ in calls] == ["kda", "projection"]
    assert calls[0][1] == (
        cuda_float32_mac_kda.patch_cuda_source,
        cuda_float32_mac_kda.EXPERIMENT_EXTENSION,
        cuda_float32_mac_kda.VARIANT,
    )
    assert calls[1][1] == (
        cuda_float32_mac_kda._float32_projection,
        cuda_float32_mac_kda._scalar_projection,
        cuda_float32_mac_kda.VARIANT,
    )
    assert event == {
        "variant": cuda_float32_mac_kda.VARIANT,
        "kda": {"event": "kda"},
        "projection": {"event": "projection"},
    }
