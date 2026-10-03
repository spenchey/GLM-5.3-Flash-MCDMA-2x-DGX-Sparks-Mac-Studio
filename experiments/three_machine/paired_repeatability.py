"""Fail-closed comparison of two clean paired-capacity campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .native_attestation import stable_configuration_digest


ROLES = ("head", "worker", "replay")
SHA256 = re.compile(r"[0-9a-f]{64}")


class RepeatabilityEvidenceError(ValueError):
    pass


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RepeatabilityEvidenceError(f"{path} is not a JSON object")
    return value


def _container(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0]
    if isinstance(value, dict):
        return value
    raise RepeatabilityEvidenceError(f"{path} is not one Docker inspect object")


def _reference_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema") != 2 or value.get("event") != "glm_concurrent_reference":
        raise RepeatabilityEvidenceError("run has no version-two frozen reference")
    result = {
        field: value.get(field)
        for field in (
            "prompt_sha256",
            "token_sha256",
            "text_sha256",
            "max_new_tokens",
            "transfer_id",
            "cache_bytes_total",
            "cache_frame_lengths",
        )
    }
    for field in ("prompt_sha256", "token_sha256", "text_sha256"):
        if not isinstance(result[field], str) or not SHA256.fullmatch(result[field]):
            raise RepeatabilityEvidenceError(f"frozen reference has invalid {field}")
    if not isinstance(result["transfer_id"], str) or not result["transfer_id"]:
        raise RepeatabilityEvidenceError("frozen reference has no transfer identity")
    if type(result["max_new_tokens"]) is not int or result["max_new_tokens"] <= 0:
        raise RepeatabilityEvidenceError("frozen reference has invalid token count")
    lengths = result["cache_frame_lengths"]
    if (
        type(result["cache_bytes_total"]) is not int
        or result["cache_bytes_total"] <= 0
        or not isinstance(lengths, list)
        or not lengths
        or any(type(item) is not int or item <= 0 for item in lengths)
        or sum(lengths) != result["cache_bytes_total"]
    ):
        raise RepeatabilityEvidenceError("frozen reference has invalid cache frame evidence")
    return result


def _accepted_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema") != 1
        or value.get("event") != "glm_paired_capacity_acceptance"
        or value.get("accepted") is not True
    ):
        raise RepeatabilityEvidenceError("run did not pass paired-capacity acceptance")
    allocation = value.get("allocation")
    if not isinstance(allocation, dict) or set(allocation) != {"spark", "mac"}:
        raise RepeatabilityEvidenceError("run has invalid allocation evidence")
    configurations = value.get("native_configuration_sha256")
    container_ids = value.get("native_container_ids")
    started_at = value.get("native_started_at")
    for label, mapping in (
        ("configuration", configurations),
        ("container", container_ids),
        ("start", started_at),
    ):
        if not isinstance(mapping, dict) or set(mapping) != set(ROLES):
            raise RepeatabilityEvidenceError(f"run has incomplete {label} identity evidence")
    for role in ROLES:
        if not isinstance(configurations[role], str) or not SHA256.fullmatch(configurations[role]):
            raise RepeatabilityEvidenceError(f"run has invalid {role} configuration hash")
        if not isinstance(container_ids[role], str) or not SHA256.fullmatch(container_ids[role]):
            raise RepeatabilityEvidenceError(f"run has invalid {role} container identity")
        if not isinstance(started_at[role], str) or not started_at[role]:
            raise RepeatabilityEvidenceError(f"run has invalid {role} start time")
    if len(set(container_ids.values())) != len(ROLES):
        raise RepeatabilityEvidenceError("one run reused a container identity across roles")
    return {
        "allocation": allocation,
        "configuration_sha256": configurations,
        "container_ids": container_ids,
        "started_at": started_at,
    }


def _configuration_identity(
    root: Path,
    acceptance: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    paths = {role: root / f"{role}-before.json" for role in ROLES}
    present = {role: path.exists() for role, path in paths.items()}
    if any(present.values()) and not all(present.values()):
        missing = ", ".join(role for role, exists in present.items() if not exists)
        raise RepeatabilityEvidenceError(
            f"run has incomplete raw configuration evidence: {missing}"
        )
    if not any(present.values()):
        return dict(acceptance["configuration_sha256"]), "acceptance"

    result: dict[str, str] = {}
    for role, path in paths.items():
        container = _container(path)
        if container.get("Id") != acceptance["container_ids"][role]:
            raise RepeatabilityEvidenceError(
                f"{role} raw configuration has the wrong container identity"
            )
        state = container.get("State") or {}
        if state.get("StartedAt") != acceptance["started_at"][role]:
            raise RepeatabilityEvidenceError(
                f"{role} raw configuration has the wrong start time"
            )
        result[role] = stable_configuration_digest(container)
    return result, "raw-docker-inspect"


def evaluate(first: Path, second: Path) -> dict[str, Any]:
    campaigns = []
    for root in (first, second):
        acceptance = _accepted_identity(_object(root / "acceptance.json"))
        configurations, source = _configuration_identity(root, acceptance)
        acceptance["configuration_sha256"] = configurations
        campaigns.append({
            "root": str(root.resolve()),
            "reference": _reference_identity(_object(root / "frozen-reference.json")),
            "acceptance": acceptance,
            "configuration_source": source,
        })
    left, right = campaigns
    if left["reference"] != right["reference"]:
        raise RepeatabilityEvidenceError("clean runs used different frozen references")
    if left["acceptance"]["allocation"] != right["acceptance"]["allocation"]:
        raise RepeatabilityEvidenceError("clean runs used different allocations")
    if (
        left["acceptance"]["configuration_sha256"]
        != right["acceptance"]["configuration_sha256"]
    ):
        raise RepeatabilityEvidenceError("clean runs used different stable configurations")
    for role in ROLES:
        if left["acceptance"]["container_ids"][role] == right["acceptance"]["container_ids"][role]:
            raise RepeatabilityEvidenceError(f"{role} container was not cleanly recreated")
        if left["acceptance"]["started_at"][role] == right["acceptance"]["started_at"][role]:
            raise RepeatabilityEvidenceError(f"{role} start time did not change across clean runs")
    return {
        "schema": 1,
        "event": "glm_paired_capacity_repeatability",
        "accepted": True,
        "runs": [left["root"], right["root"]],
        "reference": left["reference"],
        "allocation": left["acceptance"]["allocation"],
        "configuration_sha256": left["acceptance"]["configuration_sha256"],
        "configuration_sources": [
            left["configuration_source"],
            right["configuration_source"],
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.first, args.second), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
