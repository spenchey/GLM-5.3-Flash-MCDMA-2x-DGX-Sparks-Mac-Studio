"""Fail-closed identity and restart attestation for the temporary two-Spark baseline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


class NativeAttestationError(ValueError):
    pass


def _one(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise NativeAttestationError(f"{path} is not one docker-inspect record")
    return value[0]


def _health(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not (
        value.get("ok") is True or value.get("status") in {"ok", "ready"}
    ):
        raise NativeAttestationError(f"{path} does not contain ready service health")
    return value


def _configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    # Docker does not promise a stable order for the derived Mounts array.
    # Docker and the NVIDIA runtime likewise do not promise an order for a
    # unique-key Env projection. Canonicalize those set-like projections while
    # keeping every field and value covered by the configuration digest. Keep
    # duplicate environment keys in order because the last value can win.
    mounts = sorted(
        value.get("Mounts") or [],
        key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
    )
    config = copy.deepcopy(value.get("Config"))
    if isinstance(config, dict):
        environment = config.get("Env")
        if isinstance(environment, list) and all(
            isinstance(item, str) for item in environment
        ):
            keys = [item.partition("=")[0] for item in environment]
            if len(set(keys)) == len(keys):
                config["Env"] = sorted(environment)
    return {
        "Image": value.get("Image"),
        "Path": value.get("Path"),
        "Args": value.get("Args"),
        "Config": config,
        "HostConfig": value.get("HostConfig"),
        "Mounts": mounts,
    }


def _stable_configuration(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the operator-selected configuration without Docker run identity."""

    result = copy.deepcopy(_configuration(value))
    config = result.get("Config") or {}
    config.pop("Hostname", None)
    labels = config.get("Labels") or {}
    labels.pop("glm.mcdma.run", None)
    return result


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def stable_configuration_digest(value: Mapping[str, Any]) -> str:
    """Digest operator-selected settings after safe Docker canonicalization."""

    return _digest(_stable_configuration(value))


def _flag(command: Sequence[Any], name: str) -> str:
    values = [str(value) for value in command]
    if values.count(name) != 1:
        raise NativeAttestationError(f"baseline command has no unique {name} setting")
    index = values.index(name)
    if index + 1 >= len(values):
        raise NativeAttestationError(f"baseline command has no value after {name}")
    return values[index + 1]


def _validate_runtime(
    role: str,
    anchor: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    expected_image: str,
    expected_parallel: int,
    expected_model: str,
    expected_drafter: str,
) -> dict[str, Any]:
    for label, value in (("anchor", anchor), ("observed", observed)):
        if value.get("Image") != expected_image:
            raise NativeAttestationError(f"{role} {label} image differs from the exact pin")
        state = value.get("State") or {}
        if state.get("Running") is not True or state.get("OOMKilled") is not False:
            raise NativeAttestationError(f"{role} {label} container is not cleanly running")
        command = (value.get("Config") or {}).get("Cmd") or []
        if _flag(command, "--parallel") != str(expected_parallel):
            raise NativeAttestationError(f"{role} {label} has the wrong CUDA lane count")
        if _flag(command, "--backend") != "cuda" or _flag(command, "--tp") != "2":
            raise NativeAttestationError(f"{role} {label} is not the two-Spark CUDA baseline")
        if _flag(command, "--drafter") != expected_drafter:
            raise NativeAttestationError(f"{role} {label} has the wrong drafter")
        expected_rank = "0" if role == "head" else "1"
        if _flag(command, "--rank") != expected_rank:
            raise NativeAttestationError(f"{role} {label} has the wrong distributed rank")
        if expected_model not in [str(item) for item in command]:
            raise NativeAttestationError(f"{role} {label} has the wrong model path")
        labels = (value.get("Config") or {}).get("Labels") or {}
        if labels.get("glm.mcdma.mode") != "native-portable":
            raise NativeAttestationError(f"{role} {label} lacks the owned baseline label")

    if anchor.get("Id") != observed.get("Id"):
        raise NativeAttestationError(f"{role} container identity changed")
    if _digest(_configuration(anchor)) != _digest(_configuration(observed)):
        raise NativeAttestationError(f"{role} complete container configuration changed")
    anchor_state = anchor.get("State") or {}
    observed_state = observed.get("State") or {}
    if anchor_state.get("StartedAt") != observed_state.get("StartedAt"):
        raise NativeAttestationError(f"{role} container start time changed")
    if anchor.get("RestartCount") != observed.get("RestartCount"):
        raise NativeAttestationError(f"{role} container restart count changed")
    return {
        "container_id": observed.get("Id"),
        "configuration_sha256": stable_configuration_digest(observed),
        "runtime_configuration_sha256": _digest(_configuration(observed)),
        "started_at": observed_state.get("StartedAt"),
        "restart_count": observed.get("RestartCount"),
        "image_id": observed.get("Image"),
    }


def _validate_replay_runtime(
    anchor: Mapping[str, Any],
    observed: Mapping[str, Any],
    *,
    expected_image: str,
    expected_prompt: str,
) -> dict[str, Any]:
    for label, value in (("anchor", anchor), ("observed", observed)):
        if value.get("Image") != expected_image:
            raise NativeAttestationError(f"replay {label} image differs from the exact pin")
        state = value.get("State") or {}
        if state.get("Running") is not True or state.get("OOMKilled") is not False:
            raise NativeAttestationError(f"replay {label} container is not cleanly running")
        command = [str(item) for item in ((value.get("Config") or {}).get("Cmd") or [])]
        if "experiments.three_machine.cache_replay" not in command:
            raise NativeAttestationError(f"replay {label} runs the wrong module")
        labels = (value.get("Config") or {}).get("Labels") or {}
        if labels.get("glm.mcdma.mode") != "cache-replay":
            raise NativeAttestationError(f"replay {label} lacks the owned replay label")
        if labels.get("glm.mcdma.prompt") != expected_prompt:
            raise NativeAttestationError(f"replay {label} has the wrong prompt label")
    if anchor.get("Id") != observed.get("Id"):
        raise NativeAttestationError("replay container identity changed")
    if _digest(_configuration(anchor)) != _digest(_configuration(observed)):
        raise NativeAttestationError("replay complete container configuration changed")
    anchor_state = anchor.get("State") or {}
    observed_state = observed.get("State") or {}
    if anchor_state.get("StartedAt") != observed_state.get("StartedAt"):
        raise NativeAttestationError("replay container start time changed")
    if anchor.get("RestartCount") != observed.get("RestartCount"):
        raise NativeAttestationError("replay container restart count changed")
    return {
        "container_id": observed.get("Id"),
        "configuration_sha256": _digest(_stable_configuration(observed)),
        "runtime_configuration_sha256": _digest(_configuration(observed)),
        "started_at": observed_state.get("StartedAt"),
        "restart_count": observed.get("RestartCount"),
        "image_id": observed.get("Image"),
        "prompt_sha256": expected_prompt,
    }


def attest_replay(
    *,
    anchor: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    expected_image: str,
    expected_prompt: str,
) -> dict[str, Any]:
    before_result = _validate_replay_runtime(
        anchor, before, expected_image=expected_image, expected_prompt=expected_prompt
    )
    after_result = _validate_replay_runtime(
        anchor, after, expected_image=expected_image, expected_prompt=expected_prompt
    )
    if before_result != after_result:
        raise NativeAttestationError("replay identity changed inside the round")
    return after_result


def attest(
    *,
    anchor_head: Mapping[str, Any],
    anchor_worker: Mapping[str, Any],
    before_head: Mapping[str, Any],
    after_head: Mapping[str, Any],
    before_worker: Mapping[str, Any],
    after_worker: Mapping[str, Any],
    health_before: Mapping[str, Any],
    health_after: Mapping[str, Any],
    expected_image: str,
    expected_parallel: int,
    expected_model: str,
    expected_drafter: str,
) -> dict[str, Any]:
    if not (
        health_before.get("ok") is True
        or health_before.get("status") in {"ok", "ready"}
    ) or not (
        health_after.get("ok") is True
        or health_after.get("status") in {"ok", "ready"}
    ):
        raise NativeAttestationError("baseline service health changed from ready")
    head_before = _validate_runtime(
        "head", anchor_head, before_head,
        expected_image=expected_image,
        expected_parallel=expected_parallel,
        expected_model=expected_model,
        expected_drafter=expected_drafter,
    )
    head_after = _validate_runtime(
        "head", anchor_head, after_head,
        expected_image=expected_image,
        expected_parallel=expected_parallel,
        expected_model=expected_model,
        expected_drafter=expected_drafter,
    )
    worker_before = _validate_runtime(
        "worker", anchor_worker, before_worker,
        expected_image=expected_image,
        expected_parallel=expected_parallel,
        expected_model=expected_model,
        expected_drafter=expected_drafter,
    )
    worker_after = _validate_runtime(
        "worker", anchor_worker, after_worker,
        expected_image=expected_image,
        expected_parallel=expected_parallel,
        expected_model=expected_model,
        expected_drafter=expected_drafter,
    )
    if head_before != head_after or worker_before != worker_after:
        raise NativeAttestationError("baseline identity changed inside the round")
    return {
        "schema": 1,
        "accepted": True,
        "head": head_after,
        "worker": worker_after,
        "health_before": dict(health_before),
        "health_after": dict(health_after),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in (
        "anchor-head", "anchor-worker", "before-head", "after-head",
        "before-worker", "after-worker", "health-before", "health-after",
        "anchor-replay", "before-replay", "after-replay",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-image", required=True)
    parser.add_argument("--expected-parallel", type=int, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-drafter", required=True)
    parser.add_argument("--expected-prompt", required=True)
    args = parser.parse_args()
    result = attest(
        anchor_head=_one(args.anchor_head),
        anchor_worker=_one(args.anchor_worker),
        before_head=_one(args.before_head),
        after_head=_one(args.after_head),
        before_worker=_one(args.before_worker),
        after_worker=_one(args.after_worker),
        health_before=_health(args.health_before),
        health_after=_health(args.health_after),
        expected_image=args.expected_image,
        expected_parallel=args.expected_parallel,
        expected_model=args.expected_model,
        expected_drafter=args.expected_drafter,
    )
    result["replay"] = attest_replay(
        anchor=_one(args.anchor_replay),
        before=_one(args.before_replay),
        after=_one(args.after_replay),
        expected_image=args.expected_image,
        expected_prompt=args.expected_prompt,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
