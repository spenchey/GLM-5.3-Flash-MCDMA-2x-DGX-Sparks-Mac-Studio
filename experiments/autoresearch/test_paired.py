from __future__ import annotations

import copy
import hashlib
import json
from unittest.mock import patch

import pytest

from experiments.autoresearch.cli import main
from experiments.autoresearch.evaluator import ManifestError, _canonical
from experiments.autoresearch.paired import (
    BENCHMARK_SHAPES,
    CONFIRMATION_SEQUENCE,
    EQUIVALENCE_SHAPES,
    EXTENDED_SEQUENCE,
    SCREEN_SEQUENCE,
    build_round_from_evidence,
    evaluate_campaign,
    evaluate_suite,
    validate_campaign,
    validate_confirmation,
)


BASELINE = {
    "TF_GLM_EXL3_LOADS": "0",
    "TF_GLM_L2PF": "0",
    "TF_GLM_KDA_CHUNKED": "0",
    "THREE_MACHINE_PHASE_TIMING": "0",
}
CANDIDATE = {**BASELINE, "TF_GLM_L2PF": "1"}
IMAGE_ID = "sha256:" + "f" * 64


def _containers(index: int) -> dict[str, dict[str, object]]:
    result = {}
    for role, suffix in (("head", 1), ("worker", 2)):
        container_id = f"{index * 2 + suffix:064x}"
        state = {
            "name": f"/glm53-{role}", "id": container_id, "image_id": IMAGE_ID,
            "running": True, "oom_killed": False, "exit_code": 0,
        }
        result[f"{role}-before"] = copy.deepcopy(state)
        result[f"{role}-after"] = copy.deepcopy(state)
    return result


def _health(index: int, knobs: dict[str, str]) -> dict[str, object]:
    return {
        "status": "ready", "deployment_id": f"deploy-{index}", "commit": "commit-1",
        "project_manifest": "manifest-1", "config_sha256": "4" * 64,
        "image_id": IMAGE_ID, "link_generation": index + 1,
        "performance_knobs": copy.deepcopy(knobs), "failed_requests": 0,
    }


def _mcdma_state(calls: int) -> dict[str, object]:
    return {
        "link": "up", "inflight": False,
        "mac": {"alive": True, "calls": calls, "failures": 0},
        "spark": {
            "alive": True, "calls": calls, "failures": 0, "service": "poll",
        },
    }


def benchmark_bundle(
    index: int, *, arm: str = "A", decode: float | None = None,
    first_token: float | None = None, wall: float | None = None,
    final_128: float | None = None,
) -> dict[str, object]:
    knobs = copy.deepcopy(BASELINE if arm == "A" else CANDIDATE)
    decode = (1.0 if arm == "A" else 0.94) if decode is None else decode
    first_token = (1.0 if arm == "A" else 0.99) if first_token is None else first_token
    wall = (8.0 if arm == "A" else 7.92) if wall is None else wall
    final_128 = decode if final_128 is None else final_128
    health = _health(index, knobs)
    keys = [(index * 100 + run, 0) for run in range(1, 7)]
    raw_runs = [{
        "request_id": request_id,
        "steps": [{"step": step, "prompt": offset == 0}],
        "expected_mcdma_calls": 4,
        "decode_step_s": {"median": decode, "p99": decode * 1.1},
        "stall_count": 0, "hiccup_count": 0,
    } for offset, (request_id, step) in enumerate(keys)]
    benchmark = {
        "health_before": copy.deepcopy(health),
        "health_after_warmup": {
            **copy.deepcopy(health),
            "memory": {"mlx_active_bytes": 1_000_000, "mlx_cache_bytes": 1_000_000},
        },
        "health_after": copy.deepcopy(health), "warmups": raw_runs[:1], "runs": raw_runs[1:],
        "decode_step_median_s": {"median": decode},
        "first_token_s": {"median": first_token}, "wall_s": {"median": wall},
        "final_128_decode_step_median_s": {"median": final_128},
        "token_ids_sha256": "5" * 64, "reference_token_ids_sha256": "5" * 64,
        "reference_file_sha256": "3" * 64, "expected_mcdma_calls": 24,
        "mlx_post_warmup_growth_bytes": 0, "reference_id": f"shape-{index}",
    }
    events = [{"request_id": request_id, "step": step} for request_id, step in keys]
    memory = {
        "post_warmup": {"cuda_allocated_bytes": 1_000_000, "cuda_reserved_bytes": 2_000_000, "process_rss_bytes": 3_000_000},
        "final": {"cuda_allocated_bytes": 1_000_000, "cuda_reserved_bytes": 2_000_000, "process_rss_bytes": 3_000_000},
        "growth_bytes": {"cuda_allocated_bytes": 0, "cuda_reserved_bytes": 0, "process_rss_bytes": 0},
    }
    return {
        "schema": 2, "benchmark": benchmark,
        "mcdma_before": _mcdma_state(1),
        "mcdma_after": _mcdma_state(25),
        "mcdma_call_delta": {"mac": 24, "spark": 24},
        "mcdma_failure_delta": {"mac": 0, "spark": 0}, "expected_steps": len(keys),
        "spark_phase_events": copy.deepcopy(events), "rank0_commit_events": copy.deepcopy(events),
        "rank1_commit_events": copy.deepcopy(events), "joined_step_evidence": copy.deepcopy(events),
        "rank_memory": {"rank0": copy.deepcopy(memory), "rank1": copy.deepcopy(memory)},
        "host_memory": {
            "head_memavailable_drop_bytes": 0, "worker_memavailable_drop_bytes": 0,
            "head_rss_growth_bytes": 0, "worker_rss_growth_bytes": 0,
            "studio_rss_growth_bytes": 0,
        },
        "containers": _containers(index),
        "container_environment": {"head": copy.deepcopy(knobs), "worker": copy.deepcopy(knobs)},
        "measurement_code_sha256": "1" * 64,
    }


def quality_bundle(index: int, arm: str, *, fail_shape: str | None = None) -> dict[str, object]:
    knobs = copy.deepcopy(BASELINE if arm == "A" else CANDIDATE)
    health = _health(index, knobs)
    results, events = [], []
    for shape_index, shape in enumerate(EQUIVALENCE_SHAPES):
        completions = []
        for repeat in range(2):
            request_id = index * 1000 + shape_index * 10 + repeat
            observed = "6" * 64 if shape == fail_shape else "5" * 64
            completions.append({
                "request_id": request_id, "steps": [{"step": 0, "prompt": True}],
                "token_ids_sha256": observed, "reference_token_ids_sha256": "5" * 64,
            })
            events.append({"request_id": request_id, "step": 0})
        results.append({"id": shape, "completions": completions})
    expected_calls = len(events) * 4
    return {
        "schema": 2,
        "quality": {
            "health_before": copy.deepcopy(health), "health_after": copy.deepcopy(health),
            "expected_mcdma_calls": expected_calls, "results": results,
        },
        "mcdma_before": _mcdma_state(0),
        "mcdma_after": _mcdma_state(expected_calls),
        "mcdma_call_delta": {"mac": expected_calls, "spark": expected_calls},
        "mcdma_failure_delta": {"mac": 0, "spark": 0}, "expected_steps": len(events),
        "spark_phase_events": copy.deepcopy(events), "rank0_commit_events": copy.deepcopy(events),
        "rank1_commit_events": copy.deepcopy(events), "containers": _containers(index),
    }


def idle_bundle(
    index: int, arm: str, *, steady: float = 1.0,
    first: float = 1.9, second: float = 1.01,
) -> dict[str, object]:
    health = _health(index, copy.deepcopy(BASELINE if arm == "A" else CANDIDATE))
    events = [
        {"request_id": index * 1000 + request, "step": 0}
        for request in range(5)
    ]
    mcdma_before = {
        "link": "up", "inflight": None,
        "mac": {"calls": 0, "failures": 0, "alive": True},
        "spark": {"calls": 0, "failures": 0, "alive": True},
    }
    mcdma_after = {
        "link": "up", "inflight": None,
        "mac": {"calls": 20, "failures": 0, "alive": True},
        "spark": {"calls": 20, "failures": 0, "alive": True},
    }
    return {
        "schema": 1, "health_before": copy.deepcopy(health), "health_after": copy.deepcopy(health),
        "containers": _containers(index), "steady_runs_s": [steady, steady, steady],
        "idle_seconds": 3600, "first_after_idle_s": first, "second_after_idle_s": second,
        "mcdma_before": mcdma_before, "mcdma_after": mcdma_after,
        "mcdma_call_delta": {"mac": 20, "spark": 20}, "expected_mcdma_calls": 20,
        "spark_phase_events": copy.deepcopy(events),
        "rank0_commit_events": copy.deepcopy(events),
        "rank1_commit_events": copy.deepcopy(events), "expected_steps": len(events),
    }


def campaign(candidate_values=None, *, extended: bool = False, candidate_ttft: float = 0.99):
    sequence = EXTENDED_SEQUENCE if extended else SCREEN_SEQUENCE
    candidate_values = candidate_values or [0.94] * (len(sequence) // 2)
    candidates = iter(candidate_values)
    source_rounds = []
    for index, arm in enumerate(sequence):
        source_rounds.append({
            "arm": arm,
            "evidence": benchmark_bundle(
                index, arm=arm, decode=1.0 if arm == "A" else next(candidates),
                first_token=1.0 if arm == "A" else candidate_ttft,
            ),
        })
    return {
        "schema_version": 2, "campaign_id": "candidate-1",
        "baseline_settings": copy.deepcopy(BASELINE), "candidate_settings": copy.deepcopy(CANDIDATE),
        "design": {
            "primary_metric": "decode_step_median_s", "direction": "lower",
            "minimum_gain_fraction": 0.03,
            "confidence_multiplier": 2.015 if extended else 2.353,
            "max_secondary_regression_fraction": 0.02,
            "secondary_metrics": {"ttft_s": "lower"},
        },
        "baseline_precondition_evidence": [
            benchmark_bundle(100, decode=1.0), benchmark_bundle(101, decode=0.995),
            benchmark_bundle(102, decode=1.005),
        ],
        "identity": {
            "measurement_code_sha256": "1" * 64, "timer_code_sha256": "2" * 64,
            "reference_file_sha256": "3" * 64,
        },
        "source_rounds": source_rounds,
    }


def passed_screen_receipt() -> dict[str, object]:
    result = evaluate_campaign(campaign())
    assert result["verdict"] == "screen_pass"
    return result


def confirmation(screen_receipt: dict[str, object], *, mutate_round=None) -> dict[str, object]:
    source_rounds = []
    for index, arm in enumerate(CONFIRMATION_SEQUENCE):
        factor = 1.0 if arm == "A" else 0.99
        benchmarks = {
            shape: benchmark_bundle(
                index + 200, arm=arm,
                decode=(1.05 if shape == "B2" else 1.0) * factor,
                first_token=({"B3-64": 0.5, "B3-512": 1.5, "B3-2040": 4.0}.get(shape, 1.0) * factor),
                wall=8.0 * factor,
                final_128=(1.1 if shape == "B2" else 1.0) * factor,
            ) for shape in BENCHMARK_SHAPES
        }
        source = {
            "arm": arm, "benchmarks": benchmarks,
            "quality": quality_bundle(index + 200, arm),
            "idle": idle_bundle(index + 200, arm),
        }
        if mutate_round is not None:
            mutate_round(index, source)
        source_rounds.append(source)
    return {
        "schema_version": 2, "campaign_id": "candidate-1",
        "baseline_settings": copy.deepcopy(BASELINE), "candidate_settings": copy.deepcopy(CANDIDATE),
        "screen_receipt_sha256": hashlib.sha256(_canonical(screen_receipt)).hexdigest(),
        "source_rounds": source_rounds,
    }


def test_screen_pass_is_not_final_acceptance():
    result = evaluate_campaign(campaign())
    assert result["verdict"] == "screen_pass"
    assert result["final_acceptance"] is False
    assert result["lower_confidence_gain_fraction"] > 0.03


def test_hard_gate_is_derived_from_raw_evidence():
    value = campaign()
    value["source_rounds"][2]["evidence"]["benchmark"]["token_ids_sha256"] = "6" * 64
    result = evaluate_campaign(value)
    assert result["verdict"] == "reject"
    assert result["failed_gates"] == ["reference_tokens"]


def test_screen_can_extend_once_and_rejects_nonfavoring_pair():
    assert evaluate_campaign(campaign([0.94, 0.99, 0.94, 0.99]))["verdict"] == "extend_once"
    result = evaluate_campaign(campaign([0.94, 1.01, 0.94, 0.94]))
    assert result["verdict"] == "reject"
    assert "not every pair favors the candidate" in result["reasons"]


def test_twelve_round_failure_rejects_without_another_extension():
    result = evaluate_campaign(campaign([0.94, 0.94, 0.94, 0.94, 0.94, 1.01], extended=True))
    assert result["verdict"] == "reject"


def test_thresholds_and_secondary_metric_are_not_author_controlled():
    for key, value in (
        ("minimum_gain_fraction", 0.001),
        ("max_secondary_regression_fraction", 0.9),
        ("secondary_metrics", {}),
    ):
        candidate = campaign()
        candidate["design"][key] = value
        with pytest.raises(ManifestError):
            validate_campaign(candidate)


def test_campaign_rejects_drift_bad_order_reused_containers_and_typed_rounds():
    value = campaign()
    value["baseline_precondition_evidence"][1]["benchmark"]["decode_step_median_s"] = {"median": 1.03}
    with pytest.raises(ManifestError, match="more than 2%"):
        validate_campaign(value)
    value = campaign()
    value["source_rounds"][0]["arm"] = "B"
    with pytest.raises(ManifestError, match="round arms"):
        validate_campaign(value)
    value = campaign()
    value["source_rounds"][1]["evidence"]["containers"] = copy.deepcopy(
        value["source_rounds"][0]["evidence"]["containers"]
    )
    with pytest.raises(ManifestError, match="containers must be recreated"):
        validate_campaign(value)
    forged = campaign()
    forged["rounds"] = []
    with pytest.raises(ManifestError, match="keys must be exactly"):
        validate_campaign(forged)


def test_round_builder_consumes_real_wrapper_shape():
    round_ = build_round_from_evidence("B", benchmark_bundle(7, arm="B"))
    assert round_["primary"] == 0.94
    assert round_["secondary"] == {"ttft_s": 0.99}
    assert all(round_["hard_gates"].values())
    assert round_["restart_identity"]["container_ids"] == [f"{15:064x}", f"{16:064x}"]


def test_raw_benchmark_and_mcdma_evidence_fail_closed():
    malformed = benchmark_bundle(7, arm="B")
    del malformed["benchmark"]["runs"][0]["request_id"]
    with pytest.raises(ManifestError, match="malformed request evidence"):
        build_round_from_evidence("B", malformed)
    wrong_counters = benchmark_bundle(8, arm="B")
    wrong_counters["mcdma_after"]["mac"]["calls"] += 1
    with pytest.raises(ManifestError, match="call counters"):
        build_round_from_evidence("B", wrong_counters)
    wrong_environment = benchmark_bundle(9, arm="B")
    wrong_environment["container_environment"]["worker"]["TF_GLM_L2PF"] = "0"
    assert not build_round_from_evidence("B", wrong_environment)["hard_gates"][
        "container_environment"
    ]


def test_idle_evidence_requires_exact_events_and_live_link_counters():
    screen = passed_screen_receipt()
    missing_commit = confirmation(screen)
    missing_commit["source_rounds"][0]["idle"]["rank1_commit_events"].pop()
    with pytest.raises(ManifestError, match="event identities"):
        evaluate_suite(missing_commit, screen)
    down_link = confirmation(screen)
    down_link["source_rounds"][0]["idle"]["mcdma_after"]["link"] = "down"
    with pytest.raises(ManifestError, match="not up and idle"):
        evaluate_suite(down_link, screen)


def test_confirmation_suite_is_only_final_acceptance_and_is_bound_to_screen():
    screen = passed_screen_receipt()
    value = confirmation(screen)
    result = evaluate_suite(value, screen)
    assert result["verdict"] == "keep"
    assert result["final_acceptance"] is True
    rejected = copy.deepcopy(screen)
    rejected["verdict"] = "reject"
    value["screen_receipt_sha256"] = hashlib.sha256(_canonical(rejected)).hexdigest()
    with pytest.raises(ManifestError, match="passed non-final"):
        evaluate_suite(value, rejected)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda i, s: s["benchmarks"]["B2"]["benchmark"].update({"final_128_decode_step_median_s": {"median": 1.25}}) if i == 0 else None, "B2 final128"),
        (lambda i, s: s["benchmarks"]["B4"]["benchmark"].update({"wall_s": {"median": 8.5}}) if i == 3 else None, "paired regression"),
        (lambda i, s: s["benchmarks"]["B3-64"]["benchmark"].update({"first_token_s": {"median": 0.55}}) if i == 3 else None, "TTFT repeat"),
        (lambda i, s: s["idle"].update({"first_after_idle_s": 2.01}) if i == 0 else None, "idle first"),
        (lambda i, s: s["idle"].update({"second_after_idle_s": 1.03}) if i == 0 else None, "idle second"),
        (lambda i, s: s.update({"quality": quality_bundle(i + 200, s["arm"], fail_shape="E8")}) if i == 0 else None, "equivalence failed"),
    ],
)
def test_confirmation_rejects_each_suite_blocker(mutate, reason):
    screen = passed_screen_receipt()
    result = evaluate_suite(confirmation(screen, mutate_round=mutate), screen)
    assert result["verdict"] == "reject"
    assert any(reason in item for item in result["reasons"])


def test_confirmation_requires_exact_baab_order():
    screen = passed_screen_receipt()
    value = confirmation(screen)
    value["source_rounds"][0]["arm"] = "A"
    with pytest.raises(ManifestError, match="round arms must be BAAB"):
        validate_confirmation(value, screen)


def test_cli_exit_semantics(tmp_path, capsys):
    campaign_path = tmp_path / "campaign.json"
    receipts = tmp_path / "receipts"
    campaign_path.write_text(json.dumps(campaign()))
    assert main(["paired-evaluate", "--campaign", str(campaign_path), "--receipts", str(receipts)]) == 5
    screened = json.loads(capsys.readouterr().out)
    screen_result = {key: value for key, value in screened.items() if key != "receipt"}
    screen_path = tmp_path / "screen.json"
    screen_path.write_bytes(_canonical(screen_result))
    confirm_path = tmp_path / "confirmation.json"
    confirm_path.write_text(json.dumps(confirmation(screen_result)))
    assert main([
        "paired-confirm", "--campaign", str(confirm_path), "--screen-receipt", str(screen_path),
        "--receipts", str(receipts),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["final_acceptance"] is True
    wrong = copy.deepcopy(screen_result)
    wrong["verdict"] = "reject"
    screen_path.write_bytes(_canonical(wrong))
    assert main([
        "paired-confirm", "--campaign", str(confirm_path), "--screen-receipt", str(screen_path),
        "--receipts", str(receipts),
    ]) == 3
    assert capsys.readouterr().out.startswith("INVALID:")


def test_paired_evaluation_never_opens_a_network_connection():
    screen = passed_screen_receipt()
    with patch("socket.socket", side_effect=AssertionError("network attempted")):
        assert evaluate_campaign(copy.deepcopy(campaign()))["verdict"] == "screen_pass"
        assert evaluate_suite(copy.deepcopy(confirmation(screen)), screen)["verdict"] == "keep"
