import copy
import json

import pytest

from experiments.three_machine.paired_repeatability import (
    RepeatabilityEvidenceError,
    evaluate,
)


def write_campaign(root, suffix):
    root.mkdir()
    reference = {
        "schema": 2,
        "event": "glm_concurrent_reference",
        "prompt_sha256": "a" * 64,
        "token_sha256": "b" * 64,
        "text_sha256": "c" * 64,
        "max_new_tokens": 256,
        "transfer_id": "frozen-transfer",
        "cache_bytes_total": 30,
        "cache_frame_lengths": [10, 10, 10],
    }
    acceptance = {
        "schema": 1,
        "event": "glm_paired_capacity_acceptance",
        "accepted": True,
        "allocation": {"spark": 8, "mac": 8},
        "native_configuration_sha256": {
            "head": "d" * 64,
            "worker": "e" * 64,
            "replay": "f" * 64,
        },
        "native_container_ids": {
            role: (str(index) if suffix == "1" else str(index + 3)) * 64
            for index, role in enumerate(("head", "worker", "replay"), start=1)
        },
        "native_started_at": {
            role: f"2026-10-03T00:00:0{suffix}Z" for role in ("head", "worker", "replay")
        },
    }
    (root / "frozen-reference.json").write_text(json.dumps(reference))
    (root / "acceptance.json").write_text(json.dumps(acceptance))
    return reference, acceptance


def write_raw_configurations(root, acceptance, environment):
    for role in ("head", "worker", "replay"):
        container = {
            "Id": acceptance["native_container_ids"][role],
            "Image": "sha256:" + "a" * 64,
            "Path": "python3",
            "Args": ["-m", "tensorfold"],
            "Config": {
                "Hostname": f"{role}-{acceptance['native_started_at'][role]}",
                "Env": list(environment),
                "Labels": {
                    "glm.mcdma.mode": "test",
                    "glm.mcdma.run": acceptance["native_started_at"][role],
                },
            },
            "HostConfig": {"NetworkMode": "host"},
            "Mounts": [],
            "State": {"StartedAt": acceptance["native_started_at"][role]},
        }
        (root / f"{role}-before.json").write_text(json.dumps([container]))


def test_accepts_two_clean_recreated_campaigns(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_campaign(first, "1")
    write_campaign(second, "2")
    result = evaluate(first, second)
    assert result["accepted"] is True
    assert result["allocation"] == {"spark": 8, "mac": 8}


def test_rejects_changed_reference_configuration_or_reused_container(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _, first_acceptance = write_campaign(first, "1")
    second_reference, second_acceptance = write_campaign(second, "2")

    changed = copy.deepcopy(second_reference)
    changed["text_sha256"] = "9" * 64
    (second / "frozen-reference.json").write_text(json.dumps(changed))
    with pytest.raises(RepeatabilityEvidenceError, match="different frozen"):
        evaluate(first, second)

    (second / "frozen-reference.json").write_text(json.dumps(second_reference))
    changed_acceptance = copy.deepcopy(second_acceptance)
    changed_acceptance["native_configuration_sha256"]["head"] = "8" * 64
    (second / "acceptance.json").write_text(json.dumps(changed_acceptance))
    with pytest.raises(RepeatabilityEvidenceError, match="different stable configurations"):
        evaluate(first, second)

    reused = copy.deepcopy(second_acceptance)
    reused["native_container_ids"]["head"] = first_acceptance["native_container_ids"]["head"]
    (second / "acceptance.json").write_text(json.dumps(reused))
    with pytest.raises(RepeatabilityEvidenceError, match="not cleanly recreated"):
        evaluate(first, second)


def test_recanonicalizes_bound_raw_docker_environment_evidence(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _, first_acceptance = write_campaign(first, "1")
    _, second_acceptance = write_campaign(second, "2")
    second_acceptance["native_configuration_sha256"] = {
        "head": "7" * 64,
        "worker": "8" * 64,
        "replay": "9" * 64,
    }
    (second / "acceptance.json").write_text(json.dumps(second_acceptance))

    write_raw_configurations(first, first_acceptance, ["A=1", "B=2"])
    write_raw_configurations(second, second_acceptance, ["B=2", "A=1"])

    result = evaluate(first, second)
    assert result["accepted"] is True
    assert result["configuration_sources"] == [
        "raw-docker-inspect",
        "raw-docker-inspect",
    ]

    changed = json.loads((second / "head-before.json").read_text())
    changed[0]["Config"]["Env"] = ["A=1", "B=changed"]
    (second / "head-before.json").write_text(json.dumps(changed))
    with pytest.raises(RepeatabilityEvidenceError, match="different stable configurations"):
        evaluate(first, second)


def test_rejects_unbound_or_incomplete_raw_configuration_evidence(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _, first_acceptance = write_campaign(first, "1")
    write_campaign(second, "2")
    write_raw_configurations(first, first_acceptance, ["A=1"])

    raw = json.loads((first / "head-before.json").read_text())
    raw[0]["Id"] = "0" * 64
    (first / "head-before.json").write_text(json.dumps(raw))
    with pytest.raises(RepeatabilityEvidenceError, match="wrong container identity"):
        evaluate(first, second)

    (first / "head-before.json").unlink()
    with pytest.raises(RepeatabilityEvidenceError, match="incomplete raw"):
        evaluate(first, second)
