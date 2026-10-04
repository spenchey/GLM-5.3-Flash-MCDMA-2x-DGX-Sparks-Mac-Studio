import copy
import math

import pytest

from experiments.three_machine.single_stream_acceptance import (
    CONTEXT_TARGETS,
    SEQUENCE,
    SingleStreamEvidenceError,
    break_even,
    evaluate,
    evaluate_context_sweep,
)


HASHES = {
    "checkpoint_sha256": "a" * 64,
    "tokenizer_sha256": "b" * 64,
    "prompt_sha256": "c" * 64,
}


def mcdma(calls=100, failures=0):
    return {
        "link": "up",
        "generation": 7,
        "inflight": False,
        "mac": {"alive": True, "calls": calls, "failures": failures},
        "spark": {"alive": True, "calls": calls, "failures": failures},
    }


def identity(*, prompt_tokens=2048, state="warm"):
    return {
        "model_id": "TensorFold/GLM-5.3-Flash-MLX-4bit-MTP",
        "model_revision": "revision",
        **HASHES,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": 256,
        "sampling": "greedy",
        "thinking": False,
        "draft_policy": "mtp-3",
        "temperature": 0,
        "state": state,
    }


def envelope(phase, arm, run, total, *, prompt_tokens=2048, ttft=0.4, tail=0.01):
    transfer = 0.05
    calls = 0 if arm == "A" else 4
    phases = (
        {"prefill_seconds": 0.2, "decode_seconds": total - 0.2}
        if arm == "A"
        else {
            "prefill_seconds": 0.2,
            "export_seconds": 0.01,
            "transfer_seconds": transfer,
            "import_seconds": 0.01,
            "decode_seconds": total - 0.27,
        }
    )
    decode = phases["decode_seconds"]
    benchmark = {
        "schema": 1,
        "event": "glm_single_stream_round",
        "arm": arm,
        "run": run,
        "parallel": 1,
        "identity": identity(prompt_tokens=prompt_tokens),
        "completion_tokens": 256,
        "finish_reason": "length",
        "token_sha256": "d" * 64,
        "text_sha256": "e" * 64,
        "total_seconds": total,
        "ttft_seconds": ttft,
        "decode_seconds": decode,
        "decode_tokens_per_second": 255 / decode,
        "inter_token_p95_seconds": tail,
        "inter_token_p99_seconds": tail * 1.1,
        "context_window_admitted": max(131072 + 256, prompt_tokens + 256),
        "phases": phases,
        "cache": None,
    }
    if arm == "B":
        size = prompt_tokens * 4096
        benchmark["cache"] = {
            "bytes_total": size,
            "bytes_per_prompt_token": size / prompt_tokens,
            "precision": "bf16",
            "copy_count": 1,
            "complete_sparse_state": prompt_tokens <= 2048,
            "transfer_calls": calls,
            "effective_gbps": size * 8 / transfer / 1_000_000_000,
        }
    return {
        "phase": phase,
        "arm": arm,
        "ordinal": run,
        "benchmark": benchmark,
        "mcdma_before": mcdma(),
        "mcdma_after": mcdma(100 + calls),
        "safety": {"ooms": 0, "restarts": 0, "mcdma_failures": 0, "swap_bytes": 0},
    }


def campaign(candidate=0.94, *, ttft_ratio=1.0, tail_ratio=1.0):
    values = [
        envelope("warmup", "A", -2, 1.0),
        envelope("warmup", "B", -1, candidate),
    ]
    for run, arm in enumerate(SEQUENCE, start=1):
        values.append(envelope(
            "measured",
            arm,
            run,
            1.0 if arm == "A" else candidate,
            ttft=0.4 if arm == "A" else 0.4 * ttft_ratio,
            tail=0.01 if arm == "A" else 0.01 * tail_ratio,
        ))
    return values


def test_accepts_repeatable_single_stream_win():
    result = evaluate(campaign())
    assert result["accepted"] is True
    assert result["primary"]["conservative_gain_fraction"] == pytest.approx(0.06)
    assert result["mcdma_calls"] == 16
    assert result["cache"]["precision"] == "bf16"


def test_rejects_capacity_wrong_output_and_hidden_latency_regression():
    values = campaign()
    values[2]["benchmark"]["parallel"] = 8
    with pytest.raises(SingleStreamEvidenceError, match="parallel=1"):
        evaluate(values)

    values = campaign()
    values[3]["benchmark"]["token_sha256"] = "f" * 64
    with pytest.raises(SingleStreamEvidenceError, match="token IDs differ"):
        evaluate(values)

    assert evaluate(campaign(ttft_ratio=1.03))["accepted"] is False
    assert evaluate(campaign(tail_ratio=1.03))["accepted"] is False


def test_rejects_unproven_long_context_and_bad_mcdma_accounting():
    values = campaign()
    values[3]["benchmark"]["cache"]["transfer_calls"] = 3
    with pytest.raises(SingleStreamEvidenceError, match="counters differ"):
        evaluate(values)

    values = campaign()
    for item in values:
        item["benchmark"]["identity"]["prompt_tokens"] = 8192
        if item["arm"] == "B":
            size = 8192 * 4096
            item["benchmark"]["cache"].update({
                "bytes_total": size,
                "bytes_per_prompt_token": 4096,
                "effective_gbps": size * 8 / 0.05 / 1_000_000_000,
                "complete_sparse_state": False,
            })
    with pytest.raises(SingleStreamEvidenceError, match="omitted sparse"):
        evaluate(values)


def test_break_even_exposes_impossible_and_possible_compute_ceilings():
    impossible = break_even(
        prefill_seconds=0.2,
        spark_decode_tokens_per_second=70,
        mac_decode_tokens_per_second=60,
        handoff_seconds=0.05,
        output_tokens=256,
    )
    assert impossible["compute_ceiling_can_pass"] is False
    assert impossible["maximum_handoff_seconds_for_three_percent"] < 0

    possible = break_even(
        prefill_seconds=0.2,
        spark_decode_tokens_per_second=50,
        mac_decode_tokens_per_second=80,
        handoff_seconds=0.05,
        output_tokens=256,
    )
    assert possible["compute_ceiling_can_pass"] is True
    assert possible["gain_fraction"] > 0.03


def test_context_sweep_requires_cold_and_warm_accepted_results():
    results = []
    for tokens in CONTEXT_TARGETS:
        for state in ("cold", "warm"):
            results.append({
                "schema": 1,
                "event": "glm_single_stream_acceptance",
                "accepted": True,
                "identity": identity(prompt_tokens=tokens, state=state),
            })
    assert evaluate_context_sweep(results)["accepted"] is True

    missing = copy.deepcopy(results[:-1])
    with pytest.raises(SingleStreamEvidenceError, match="differs"):
        evaluate_context_sweep(missing)

    failed = copy.deepcopy(results)
    failed[-1]["accepted"] = False
    with pytest.raises(SingleStreamEvidenceError, match="did not pass"):
        evaluate_context_sweep(failed)
