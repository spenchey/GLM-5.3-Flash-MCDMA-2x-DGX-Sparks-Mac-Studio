from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.three_machine.cuda_torch_qmm import (
    install_torch_projection_variant,
)


def test_variant_changes_only_layer_zero_front_projection() -> None:
    original_calls = []
    mm_calls = []
    projection_calls = []

    def original(layer, buffers, lo, hi):
        original_calls.append((layer.index, lo, hi))

    def mm(*args):
        mm_calls.append(args)

    def projector(x, quantized, out):
        projection_calls.append((x.copy(), quantized))
        out[...] = 7

    forward = SimpleNamespace(kda_front=original, mm=mm)
    event = install_torch_projection_variant(
        forward_module=forward,
        projector=projector,
        emit=False,
    )
    buffers = SimpleNamespace(
        normed=np.arange(12).reshape(3, 4),
        kproj=np.zeros((1, 3, 300), dtype=np.float32),
        ka=np.zeros((3, 8), dtype=np.float32),
        kg=np.zeros((3, 8), dtype=np.float32),
    )
    kda = SimpleNamespace(proj="wide", fb="forget", gb="gate", fa_off=4, ga_off=132)

    forward.kda_front(SimpleNamespace(index=1, kda=kda), buffers, 0, 2)
    forward.kda_front(SimpleNamespace(index=0, kda=kda), buffers, 1, 3)

    assert event == {
        "event": "glm_cuda_projection_variant",
        "layer": 0,
        "variant": "torch-qmm-layer0",
    }
    assert original_calls == [(1, 0, 2)]
    assert len(projection_calls) == 1
    assert projection_calls[0][1] == "wide"
    assert projection_calls[0][0].tolist() == buffers.normed[1:3].tolist()
    assert np.all(buffers.kproj[0, 1:3] == 7)
    assert len(mm_calls) == 2
    assert mm_calls[0][2] == "forget"
    assert mm_calls[1][2] == "gate"


def test_variant_refuses_double_install() -> None:
    forward = SimpleNamespace(kda_front=lambda *_args: None, mm=lambda *_args: None)
    install_torch_projection_variant(
        forward_module=forward,
        projector=lambda *_args: None,
        emit=False,
    )
    with pytest.raises(RuntimeError, match="already installed"):
        install_torch_projection_variant(
            forward_module=forward,
            projector=lambda *_args: None,
            emit=False,
        )


def test_variant_name_can_identify_a_diagnostic_backend() -> None:
    forward = SimpleNamespace(kda_front=lambda *_args: None, mm=lambda *_args: None)

    event = install_torch_projection_variant(
        forward_module=forward,
        projector=lambda *_args: None,
        variant="scalar-qmm-layer0",
        emit=False,
    )

    assert event["variant"] == "scalar-qmm-layer0"


def test_auxiliary_projector_can_replace_layer_zero_follow_on_projections() -> None:
    original_calls = []
    projection_calls = []

    def original(layer, buffers, lo, hi):
        original_calls.append((layer.index, lo, hi))

    def projector(x, quantized, out):
        projection_calls.append((x.shape, quantized))
        out[...] = len(projection_calls)

    forward = SimpleNamespace(kda_front=original, mm=lambda *_args: None)
    install_torch_projection_variant(
        forward_module=forward,
        projector=projector,
        auxiliary_projector=projector,
        emit=False,
    )
    buffers = SimpleNamespace(
        normed=np.arange(12).reshape(3, 4),
        kproj=np.zeros((1, 3, 300), dtype=np.float32),
        ka=np.zeros((3, 8), dtype=np.float32),
        kg=np.zeros((3, 8), dtype=np.float32),
    )
    kda = SimpleNamespace(proj="wide", fb="forget", gb="gate", fa_off=4, ga_off=132)

    forward.kda_front(SimpleNamespace(index=0, kda=kda), buffers, 1, 3)

    assert original_calls == []
    assert projection_calls == [((2, 4), "wide"), ((2, 128), "forget"), ((2, 128), "gate")]
    assert np.all(buffers.kproj[0, 1:3] == 1)
    assert np.all(buffers.ka[1:3] == 2)
    assert np.all(buffers.kg[1:3] == 3)
