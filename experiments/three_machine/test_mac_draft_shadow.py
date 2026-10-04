from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from experiments.three_machine.mac_draft_shadow import _classes


def test_shadow_slot_copy_owns_the_nested_mtp_cache():
    slot_class, _runtime_class = _classes()
    slot = slot_class(SimpleNamespace(), chains=True)
    slot.mtp_cache.keys = mx.arange(12).reshape(3, 4)
    slot.mtp_cache.offset = 3
    slot.mtp_ready = True

    cloned = copy.copy(slot)
    mx.eval(cloned.mtp_cache.keys)

    assert cloned is not slot
    assert cloned.mtp_cache is not slot.mtp_cache
    assert cloned.mtp_cache.offset == 3
    assert cloned.mtp_ready is True
    assert bool(mx.array_equal(cloned.mtp_cache.keys, slot.mtp_cache.keys))


def test_shadow_records_complementary_first_token_hits():
    slot_class, runtime_class = _classes()
    runtime = runtime_class.__new__(runtime_class)
    runtime.reset_shadow()
    slot = slot_class(SimpleNamespace())

    slot.mtp_pending = 17
    runtime._record_previous(slot, [17], [0])
    slot.mtp_pending = 18
    runtime._record_previous(slot, [17], [0, 1])
    slot.mtp_pending = 17
    runtime._record_previous(slot, [17], [0, 1])
    slot.mtp_pending = 18
    runtime._record_previous(slot, [17], [0])

    assert runtime.shadow_totals == {
        "observations": 4,
        "both_hit": 1,
        "mtp_only": 1,
        "dflash_only": 1,
        "neither": 1,
        "mtp_predictions": 0,
    }
