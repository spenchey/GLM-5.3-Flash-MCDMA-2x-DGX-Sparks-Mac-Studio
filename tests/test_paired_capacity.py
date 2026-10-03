import copy
from pathlib import Path

import pytest

from experiments.three_machine.paired_capacity import (
    CapacityEvidenceError,
    SEQUENCE,
    evaluate,
)


HASH = "a" * 64
TEXT_HASH = "b" * 64
PROMPT_HASH = "c" * 64
TOKENS = 256
FRAME_LENGTHS = [64 * 1024 * 1024, 64 * 1024 * 1024, 20 * 1024 * 1024]
CACHE_BYTES = sum(FRAME_LENGTHS)
TRANSFER_ID = "transfer-frozen"


def native_attestation():
    def peer(role):
        return {
            "container_id": f"{role}-container",
            "configuration_sha256": ("d" if role == "head" else "e") * 64,
            "runtime_configuration_sha256": ("1" if role == "head" else "2") * 64,
            "started_at": "2026-10-02T00:00:00Z",
            "restart_count": 0,
            "image_id": "sha256:" + "f" * 64,
        }
    return {
        "schema": 1,
        "accepted": True,
        "head": peer("head"),
        "worker": peer("worker"),
        "replay": {
            **peer("replay"),
            "prompt_sha256": PROMPT_HASH,
        },
        "health_before": {"status": "ready"},
        "health_after": {"status": "ready"},
    }


def mcdma(calls, failures=0):
    return {
        "generation": 1,
        "link": "up",
        "inflight": False,
        "reply_present": False,
        "reply_transport": "rdma-read-pull",
        "mac": {"alive": True, "calls": calls, "failures": failures, "pid": 101},
        "spark": {
            "alive": True, "calls": calls, "failures": failures, "pid": 202,
            "service": "poll",
        },
    }


def benchmark(arm, run, rate, latency=1.0):
    parallel = 2
    lane = {
        "tokens": TOKENS,
        "token_sha256": HASH,
        "text_sha256": TEXT_HASH,
        "first_token_seconds": latency,
        "completion_seconds": latency,
    }
    lanes = []
    for kind, count in (("spark", 2 if arm == "A" else 1), ("mac", 0 if arm == "A" else 1)):
        for index in range(count):
            item = copy.deepcopy(lane)
            item.update({"kind": kind, "lane": index})
            lanes.append(item)
    value = {
        "schema": 2,
        "event": "glm_paired_capacity_round",
        "arm": arm,
        "run": run,
        "parallel": parallel,
        "allocation": {"spark": 2, "mac": 0} if arm == "A" else {"spark": 1, "mac": 1},
        "completion_tokens": 2 * TOKENS,
        "max_new_tokens": TOKENS,
        "seconds": (2 * TOKENS) / rate,
        "aggregate_pipeline_tokens_per_second": rate,
        "max_ttft_seconds": latency,
        "p95_completion_seconds": latency,
        "max_completion_seconds": latency,
        "token_sha256": HASH,
        "text_sha256": TEXT_HASH,
        "prompt_sha256": PROMPT_HASH,
        "lanes": lanes,
    }
    if arm == "B":
        value["mac"] = {
            "cache_wire_transfers": 1,
            "cache_reuse_copies": 0,
            "transfer_id": TRANSFER_ID,
            "prompt_sha256": PROMPT_HASH,
            "cache_bytes_total": CACHE_BYTES,
            "cache_frame_count": 3,
            "cache_frame_lengths": list(FRAME_LENGTHS),
        }
    return value


def envelope(phase, arm, run, rate, *, before=100, delta=None, latency=1.0):
    delta = (0 if arm == "A" else 5) if delta is None else delta
    return {
        "phase": phase,
        "arm": arm,
        "ordinal": run,
        "benchmark": benchmark(arm, run, rate, latency),
        "mcdma_before": mcdma(before),
        "mcdma_after": mcdma(before + delta),
        "native_attestation": native_attestation(),
    }


def passing_evidence(candidate_rate=104.0, candidate_latency=1.01):
    values = []
    counter = 100
    values.append(envelope("warmup", "A", -2, 100.0, before=counter))
    values.append(envelope("warmup", "B", -1, candidate_rate, before=counter))
    counter += 5
    for run, arm in enumerate(SEQUENCE, start=1):
        values.append(envelope(
            "measured",
            arm,
            run,
            100.0 if arm == "A" else candidate_rate,
            before=counter,
            latency=1.0 if arm == "A" else candidate_latency,
        ))
        if arm == "B":
            counter += 5
    return values


def evaluate_values(values):
    return evaluate(
        values,
        parallel=2,
        mac_parallel=1,
        prompt_sha256=PROMPT_HASH,
        token_sha256=HASH,
        text_sha256=TEXT_HASH,
        max_new_tokens=TOKENS,
        expected_transfer_id=TRANSFER_ID,
        expected_cache_bytes=CACHE_BYTES,
        expected_frame_lengths=FRAME_LENGTHS,
        preparation_timings={
            "reference_wall_seconds": 30.0,
            "cuda_prefill_head_seconds": 2.0,
            "cuda_prefill_worker_seconds": 2.1,
            "mcdma_transfer_seconds": 1.0,
            "mac_import_seconds": 0.5,
        },
        expected_mcdma_calls=5,
    )


def test_accepts_every_pair_with_confident_gain_and_bounded_tail():
    result = evaluate_values(passing_evidence())
    assert result["accepted"] is True
    assert result["sequence"] == "ABBABAAB"
    assert result["candidate_mcdma_calls_per_round"] == 5
    assert result["lower_confidence_gain_fraction"] == pytest.approx(0.04)
    assert result["median_paired_gain_fraction"] == pytest.approx(0.04)
    assert all(pair["throughput_accepted"] for pair in result["pairs"])


def test_rejects_one_slow_pair_even_if_the_median_wins():
    values = passing_evidence(candidate_rate=110.0)
    candidate = next(
        value for value in values
        if value["phase"] == "measured" and value["arm"] == "B"
    )
    candidate["benchmark"]["aggregate_pipeline_tokens_per_second"] = 102.0
    candidate["benchmark"]["seconds"] = (2 * TOKENS) / 102.0
    result = evaluate_values(values)
    assert result["accepted"] is False
    assert any("pair 1 throughput" in reason for reason in result["reasons"])


def test_rejects_summary_metrics_not_derived_from_raw_lane_evidence():
    values = passing_evidence()
    values[0]["benchmark"]["aggregate_pipeline_tokens_per_second"] = 999.0
    with pytest.raises(CapacityEvidenceError, match="differs from raw lane evidence"):
        evaluate_values(values)


def test_rejects_lane_topology_not_matching_declared_allocation():
    values = passing_evidence()
    values[1]["benchmark"]["lanes"][1].update({"kind": "spark", "lane": 1})
    with pytest.raises(CapacityEvidenceError, match="lane topology"):
        evaluate_values(values)


def test_rejects_ttft_or_tail_regression_over_two_percent():
    result = evaluate_values(passing_evidence(candidate_latency=1.021))
    assert result["accepted"] is False
    assert any("max_ttft_seconds" in reason for reason in result["reasons"])
    assert any("p95_completion_seconds" in reason for reason in result["reasons"])


def test_rejects_any_baseline_mcdma_call():
    values = passing_evidence()
    values[0]["mcdma_after"] = mcdma(101)
    with pytest.raises(CapacityEvidenceError, match="baseline made an MCDMA call"):
        evaluate_values(values)


def test_rejects_candidate_call_count_other_than_exact_contract():
    values = passing_evidence()
    values[1]["mcdma_after"] = mcdma(104)
    for value in values[2:]:
        before_calls = value["mcdma_before"]["mac"]["calls"] - 1
        after_calls = value["mcdma_after"]["mac"]["calls"] - 1
        value["mcdma_before"] = mcdma(before_calls)
        value["mcdma_after"] = mcdma(after_calls)
    with pytest.raises(CapacityEvidenceError, match="expected exactly 5"):
        evaluate_values(values)


def test_rejects_attempt_to_loosen_call_contract():
    values = passing_evidence()
    with pytest.raises(CapacityEvidenceError, match="frame count plus"):
        evaluate(
            values,
            parallel=2,
            mac_parallel=1,
            prompt_sha256=PROMPT_HASH,
            token_sha256=HASH,
            text_sha256=TEXT_HASH,
            max_new_tokens=TOKENS,
            expected_transfer_id=TRANSFER_ID,
            expected_cache_bytes=CACHE_BYTES,
            expected_frame_lengths=FRAME_LENGTHS,
            preparation_timings={
                "reference_wall_seconds": 30.0,
                "cuda_prefill_head_seconds": 2.0,
                "cuda_prefill_worker_seconds": 2.1,
                "mcdma_transfer_seconds": 1.0,
                "mac_import_seconds": 0.5,
            },
            expected_mcdma_calls=6,
        )


def test_rejects_wrong_sequence_and_output_mismatch():
    values = passing_evidence()
    values[2], values[3] = values[3], values[2]
    with pytest.raises(CapacityEvidenceError, match="schedule"):
        evaluate_values(values)

    values = passing_evidence()
    values[3], values[4] = values[4], values[3]
    with pytest.raises(CapacityEvidenceError, match="schedule"):
        evaluate_values(values)


def test_rejects_wrong_frame_manifest_token_count_and_restart():
    values = passing_evidence()
    values[1]["benchmark"]["mac"]["cache_frame_lengths"][2] -= 1
    with pytest.raises(CapacityEvidenceError, match="frame evidence"):
        evaluate_values(values)

    values = passing_evidence()
    values[1]["benchmark"]["lanes"][0]["tokens"] = TOKENS - 1
    with pytest.raises(CapacityEvidenceError, match="fixed reply length"):
        evaluate_values(values)

    values = passing_evidence()
    values[1]["native_attestation"]["head"]["restart_count"] = 1
    with pytest.raises(CapacityEvidenceError, match="restart count"):
        evaluate_values(values)

    values = passing_evidence()
    values[-1]["benchmark"]["lanes"][0]["token_sha256"] = "c" * 64
    with pytest.raises(CapacityEvidenceError, match="lane token hash"):
        evaluate_values(values)


def test_rejects_cache_identity_bytes_or_transfer_mismatch():
    values = passing_evidence()
    values[1]["benchmark"]["mac"]["transfer_id"] = "different"
    with pytest.raises(CapacityEvidenceError, match="transfer identity"):
        evaluate_values(values)

    values = passing_evidence()
    values[1]["benchmark"]["mac"]["cache_bytes_total"] += 1
    with pytest.raises(CapacityEvidenceError, match="byte count"):
        evaluate_values(values)


def test_rejects_mcdma_discontinuity_restart_or_detached_service():
    values = passing_evidence()
    values[2]["mcdma_before"] = mcdma(999)
    with pytest.raises(CapacityEvidenceError, match="discontinuous"):
        evaluate_values(values)

    values = passing_evidence()
    values[1]["mcdma_after"]["spark"]["pid"] = 303
    with pytest.raises(CapacityEvidenceError, match="daemon changed"):
        evaluate_values(values)

    values = passing_evidence()
    values[0]["mcdma_before"]["spark"]["service"] = "none"
    with pytest.raises(CapacityEvidenceError, match="service was not attached"):
        evaluate_values(values)


def test_rejects_replay_or_one_native_role_changing_across_rounds():
    values = passing_evidence()
    values[-1]["native_attestation"]["replay"]["container_id"] = "new-replay"
    with pytest.raises(CapacityEvidenceError, match="replay container identity"):
        evaluate_values(values)

    values = passing_evidence()
    values[-1]["native_attestation"]["head"]["container_id"] = "new-head"
    with pytest.raises(CapacityEvidenceError, match="head container identity"):
        evaluate_values(values)


def test_live_wrapper_defaults_to_final_sixteen_request_eight_plus_eight_gate():
    script = Path("scripts/benchmark-concurrent-capacity.sh").read_text()
    assert "total_parallel=${2:-16}" in script
    assert "mac_parallel=${3:-8}" in script
    assert "expected_mcdma_calls=$((expected_frame_count + 2))" in script
    assert "sequence=ABBABAAB" in script
    assert "experiments.three_machine.native_attestation" in script
    assert "stop-concurrent-worker.py" in script
    assert "acquire_three_machine_lifecycle_lock" in script
    assert "TENSORFOLD_NO_UPDATE_CHECK=1" in script
    assert "tr '[:upper:]' '[:lower:]'" in script
    assert "replay-before.json" in script
    assert "--expected-prompt \"$digest\"" in script
    assert 'export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"' in script

    prepare = Path("scripts/prepare-concurrent-cache.sh").read_text()
    assert '"schema": 2' in prepare
    assert '"transfer_id": transfer_id' in prepare
    assert '"preparation_timings"' in prepare
    assert '"cuda_prefill_head_seconds"' in prepare
    assert '"mcdma_transfer_seconds"' in prepare
