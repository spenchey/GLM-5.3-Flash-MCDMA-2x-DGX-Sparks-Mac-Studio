import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark-evidence.py"
SPEC = importlib.util.spec_from_file_location("benchmark_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def _event(name: str, request_id: int, step: int, **extra):
    value = {"event": name, "request_id": request_id, "step": step}
    value.update(extra)
    return json.dumps(value, separators=(",", ":"))


def _rank_event(name: str, request_id: int, step: int, *, rss: int = 1_000_000):
    return _event(
        name,
        request_id,
        step,
        cuda_allocated_bytes=2_000_000,
        cuda_reserved_bytes=3_000_000,
        process_rss_bytes=rss,
    )


def _write_bundle(path: Path, *, timing: str = "0", rss_final: int = 1_000_000):
    run = lambda request_id: {
        "request_id": request_id,
        "expected_mcdma_calls": 4,
        "steps": [{
            "step": 0,
            "prompt": True,
            "spark_mcdma_roundtrip_s": 0.031,
        }],
    }
    benchmark = {
        "health_before": {"performance_knobs": {"THREE_MACHINE_PHASE_TIMING": timing}},
        "warmups": [run(11)],
        "runs": [run(22)],
        "expected_mcdma_calls": 8,
    }
    (path / "benchmark.json").write_text(json.dumps(benchmark))
    link = {
        "link": "up",
        "inflight": False,
        "mac": {"alive": True, "failures": 0, "calls": 10},
        "spark": {"alive": True, "failures": 0, "calls": 20, "service": "poll"},
    }
    (path / "link-before.json").write_text(json.dumps(link))
    link["mac"]["calls"] += 8
    link["spark"]["calls"] += 8
    (path / "link-after.json").write_text(json.dumps(link))
    phase = {
        "timing_mode": "synchronized" if timing == "1" else "host-enqueue",
        "gpu_total_s": 0.025,
        "spark_service_s": 0.029,
    }
    if timing == "1":
        phase.update({field: 0.001 for field in evidence.PHASE_FIELDS})
    head = [
        _event("three_machine_step", 11, 0, **phase),
        _rank_event("three_machine_rank0_commit", 11, 0),
        _event("three_machine_step", 22, 0, **phase),
        _rank_event("three_machine_rank0_commit", 22, 0, rss=rss_final),
    ]
    worker = [
        _rank_event("three_machine_rank1_commit", 11, 0),
        _rank_event("three_machine_rank1_commit", 22, 0, rss=rss_final),
    ]
    (path / "head-runtime.log").write_text("\n".join(head))
    (path / "worker-runtime.log").write_text("\n".join(worker))
    host = "MemAvailable: 100000000 kB\nVmRSS: 1000000 kB\n"
    for name in ("head", "worker"):
        (path / f"{name}-host-before.txt").write_text(host)
        (path / f"{name}-host-after.txt").write_text(host)
    studio = "process_rss_kib=1000000\n"
    (path / "studio-host-before.txt").write_text(studio)
    (path / "studio-host-after.txt").write_text(studio)
    image = "sha256:" + "f" * 64
    for role, container_id in (("head", "a" * 64), ("worker", "b" * 64)):
        state = f"/glm53|{container_id}|{image}|true|false|0"
        (path / f"{role}-before.txt").write_text(state)
        (path / f"{role}-after.txt").write_text(state)
        (path / f"{role}-container-env.txt").write_text(
            "THREE_MACHINE_PHASE_TIMING=" + timing
        )
    (path / "measurement-code-sha256.txt").write_text("a" * 64)


def test_bundle_joins_steps_and_gates_rank_memory(tmp_path):
    _write_bundle(tmp_path)
    result = evidence.validate_bundle(tmp_path)
    assert result["schema"] == 2
    assert result["joined_step_evidence"][0]["transport_wake_residual_s"] == pytest.approx(0.002)
    assert result["rank_memory"]["rank0"]["growth_bytes"]["process_rss_bytes"] == 0
    assert result["mcdma_call_delta"] == {"mac": 8, "spark": 8}


def test_bundle_rejects_host_enqueue_values_labeled_as_synchronized_phases(tmp_path):
    _write_bundle(tmp_path)
    log = tmp_path / "head-runtime.log"
    lines = log.read_text().splitlines()
    value = json.loads(lines[0])
    value["layers_s"] = 0.02
    lines[0] = json.dumps(value)
    log.write_text("\n".join(lines))
    with pytest.raises(ValueError, match="falsely labels"):
        evidence.validate_bundle(tmp_path)


def test_bundle_rejects_rank_memory_growth(tmp_path):
    _write_bundle(tmp_path, rss_final=80_000_000)
    with pytest.raises(ValueError, match="rank 0 process_rss_bytes grew"):
        evidence.validate_bundle(tmp_path)


def test_bundle_requires_one_exact_spark_event_for_each_mac_step(tmp_path):
    _write_bundle(tmp_path)
    log = tmp_path / "head-runtime.log"
    log.write_text("\n".join(
        line for line in log.read_text().splitlines()
        if not ('\"event\":\"three_machine_step\"' in line and '\"request_id\":22' in line)
    ))
    with pytest.raises(ValueError, match="head Spark steps emitted 1 events"):
        evidence.validate_bundle(tmp_path)
