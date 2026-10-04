from __future__ import annotations

import pytest

from experiments.three_machine.mac_draft_selector import AdaptiveDraftChoice


def test_choice_explores_both_then_keeps_the_faster_arm():
    choice = AdaptiveDraftChoice(first="f", explore=2, every=0)

    assert choice.pick() == "f"
    choice.record("f", 2, 10.0)
    assert choice.pick() == "f"
    choice.record("f", 2, 10.0)
    assert choice.pick() == "m"
    choice.record("m", 1, 10.0)
    assert choice.pick() == "m"
    choice.record("m", 1, 10.0)

    assert choice.pick() == "f"
    assert choice.rate("f") == pytest.approx(0.2)
    assert choice.rate("m") == pytest.approx(0.1)


def test_choice_rejects_invalid_observations():
    choice = AdaptiveDraftChoice()

    with pytest.raises(ValueError):
        choice.record("x", 1, 1.0)
    with pytest.raises(ValueError):
        choice.record("f", 0, 1.0)
    with pytest.raises(ValueError):
        choice.record("f", 1, 0.0)
