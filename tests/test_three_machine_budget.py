import pytest

from experiments.three_machine.mac_stage import MetalStage, dense_context_capacity, validate_token_budget


def test_token_budget_accounts_for_first_token_from_prompt_window():
    validate_token_budget([1, 2], 2050)
    with pytest.raises(ValueError, match="needs 2052 positions"):
        validate_token_budget([1, 2], 2051)


def test_token_budget_rejects_empty_prompt():
    with pytest.raises(ValueError, match="zero tokens"):
        validate_token_budget([], 1)


def test_dense_capacity_matches_the_pinned_cuda_engine_formula():
    assert dense_context_capacity(2048, 4) == 2051


def test_metal_commit_rolls_back_only_rejected_rows():
    stage = MetalStage.__new__(MetalStage)
    calls = []
    stage.model = type("Model", (), {"keep_rows": lambda _self, cache, rows, keep: calls.append((cache, rows, keep))})()
    stage.cache = [object()]
    stage.pending = (4, True)

    stage.commit(2)

    assert stage.pending is None
    assert calls == [(stage.cache, 4, 2)]

    stage.pending = (4, True)
    stage.commit(4)

    assert stage.pending is None
    assert calls == [(stage.cache, 4, 2)]
