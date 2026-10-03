#!/usr/bin/env python3
"""Capture repeatable greedy token hashes from the pinned native two-Spark engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from pathlib import Path
from typing import Any


WORDS = (
    "time year people way day thing life world school state group country problem hand place case week "
    "company system program question work number point home water room area story fact study book job word "
    "business issue side kind head service friend power hour game line end member law city community name "
    "team minute idea body information office health person art history result change reason research river "
    "mountain signal engine garden theory market winter method bridge letter window voice paper field"
).split()

NATIVE_SOURCE_KIND = "native-two-spark"


def valid_sha256(value: Any, *, prefixed: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    candidate = value.removeprefix("sha256:") if prefixed else value
    return len(candidate) == 64 and all(char in "0123456789abcdef" for char in candidate)


def valid_git_oid(value: Any) -> bool:
    return isinstance(value, str) and len(value) in {40, 64} \
        and all(char in "0123456789abcdef" for char in value)


def health_source(
    health: dict[str, Any], model: str, attestation: dict[str, Any],
    expected_image_id: str,
) -> dict[str, Any]:
    """Combine endpoint liveness with independently captured container identity."""
    current_health = health.get("ok") is True and health.get("backend") == "tensorfold"
    legacy_health = health.get("status") in {"ok", "ready"}
    if not (current_health or legacy_health):
        raise RuntimeError("native endpoint health is not ready")
    if legacy_health and health.get("model") != model \
            and model not in (health.get("model_ids") or []):
        raise RuntimeError("native endpoint health does not attest the requested model")

    if not valid_sha256(expected_image_id, prefixed=True) \
            or not expected_image_id.startswith("sha256:"):
        raise RuntimeError("expected native image must be a full sha256 image ID")
    required = {
        "schema", "kind", "model", "image_id", "drafter", "recipe_commit",
        "recipe_tree", "recipe_config_sha256", "head", "worker",
    }
    if set(attestation) != required or attestation.get("schema") != 1:
        raise RuntimeError("native container attestation has an unexpected schema")
    if attestation.get("kind") != NATIVE_SOURCE_KIND or attestation.get("model") != model:
        raise RuntimeError("native container attestation identifies the wrong source")
    if attestation.get("image_id") != expected_image_id:
        raise RuntimeError("native container attestation image is not the pinned image")
    if attestation.get("drafter") != "mtp":
        raise RuntimeError("native container attestation does not prove the MTP drafter")
    for name in ("recipe_commit", "recipe_tree"):
        if not valid_git_oid(attestation.get(name)):
            raise RuntimeError(f"native container attestation has no valid {name}")
    if not valid_sha256(attestation.get("recipe_config_sha256")):
        raise RuntimeError("native container attestation has no valid recipe_config_sha256")
    container_ids = []
    for role in ("head", "worker"):
        container = attestation.get(role)
        if not isinstance(container, dict) or set(container) != {
            "name", "container_id", "image_id", "running", "oom_killed", "exit_code",
        }:
            raise RuntimeError(f"native {role} container attestation is malformed")
        if container.get("image_id") != expected_image_id or container.get("running") is not True \
                or container.get("oom_killed") is not False or container.get("exit_code") != 0:
            raise RuntimeError(f"native {role} container is not cleanly running the pinned image")
        container_id = container.get("container_id")
        if not valid_sha256(container_id):
            raise RuntimeError(f"native {role} container ID is malformed")
        container_ids.append(container_id)
    if len(set(container_ids)) != 2:
        raise RuntimeError("native head and worker container IDs are not distinct")
    return {
        "kind": NATIVE_SOURCE_KIND,
        "model": model,
        "image_id": expected_image_id,
        "drafter": "mtp",
        "recipe_commit": attestation["recipe_commit"],
        "recipe_tree": attestation["recipe_tree"],
        "recipe_config_sha256": attestation["recipe_config_sha256"],
        "containers": {role: attestation[role] for role in ("head", "worker")},
    }


def health_evidence(health: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        "status": health.get("status", "ok" if health.get("ok") is True else None),
        "backend": health.get("backend"),
        "model": health.get("model"),
        "model_ids": health.get("model_ids"),
        "source_kind": source["kind"],
        "image_id": source["image_id"],
        "drafter": source["drafter"],
    }
    for name in ("tensor_parallel_size", "world_size", "workers"):
        if name in health:
            evidence[name] = health[name]
    return evidence


def ensure_output_path(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"refusing to overwrite existing reference file: {path}; pass --force deliberately")


def write_output(path: Path, value: dict[str, Any], force: bool) -> None:
    mode = "w" if force else "x"
    try:
        with path.open(mode) as output:
            json.dump(value, output, indent=2, ensure_ascii=False, sort_keys=True)
            output.write("\n")
    except FileExistsError as error:
        raise SystemExit(
            f"refusing to overwrite existing reference file: {path}; pass --force deliberately"
        ) from error


def post(url: str, value: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        json.dumps(value, separators=(",", ":")).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def prose_words(seed: int, count: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) + ("." if index % 13 == 12 else "") for index in range(count))


def prompt_tokens(base_url: str, model: str, prompt: str, timeout: float) -> list[int]:
    result = post(
        base_url + "/tokenize",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout,
    )
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or any(type(value) is not int for value in tokens):
        raise RuntimeError("native /tokenize returned no integer token list")
    return tokens


def sized_prompt(base_url: str, model: str, target: int, seed: int, timeout: float) -> tuple[str, int]:
    if not 1 <= target <= 2040:
        raise ValueError("target prompt tokens must be 1..2040")
    pool = prose_words(seed, target * 2)
    words = pool.split()
    low, high = 1, len(words)
    best: tuple[str, int] | None = None
    while low <= high:
        middle = (low + high) // 2
        prompt = "Read this context, then answer with only the word ready. " + " ".join(words[:middle])
        count = len(prompt_tokens(base_url, model, prompt, timeout))
        if count <= target:
            best = (prompt, count)
            low = middle + 1
        else:
            high = middle - 1
    if best is None or best[1] != target:
        raise RuntimeError(f"could not construct an exact {target}-token native prompt")
    return best


def token_ids(response: dict[str, Any]) -> list[int]:
    values = (response.get("tensorfold") or {}).get("token_ids")
    if values is None:
        values = response["choices"][0].get("token_ids")
    if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
        raise RuntimeError("native completion returned no token IDs")
    return values


def validation_spec(task: dict[str, Any]) -> tuple[str, Any]:
    validator = task.get("validator")
    expected = task.get("expected")
    if validator is None and "expected_symbols" in task:
        validator = "python_symbols"
        expected = task["expected_symbols"]
    if validator in {"python_symbols", "contains_all"}:
        if not isinstance(expected, list) or not expected \
                or any(not isinstance(item, str) or not item for item in expected):
            raise ValueError(f"{validator} requires a non-empty string list in expected")
    elif validator == "exact_json":
        if "expected" not in task:
            raise ValueError("exact_json requires expected")
    elif validator == "integer":
        if type(expected) is not int:
            raise ValueError("integer requires an integer expected value")
    elif validator == "min_words":
        if type(expected) is not int or expected < 1:
            raise ValueError("min_words requires a positive integer expected value")
    else:
        raise ValueError(f"unsupported response validator: {validator!r}")
    return str(validator), expected


def capture(base_url: str, model: str, prompt: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "draft": False,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    responses = [post(base_url + "/v1/chat/completions", body, timeout) for _ in range(2)]
    for response in responses:
        if response.get("model") != model:
            raise RuntimeError("native completion does not attest the requested model")
        runtime = response.get("tensorfold")
        if not isinstance(runtime, dict) or runtime.get("drafts") is not False:
            raise RuntimeError("native completion does not prove MTP drafts were disabled")
    ids = [token_ids(response) for response in responses]
    if ids[0] != ids[1]:
        raise RuntimeError("native greedy token IDs changed across two captures")
    outputs = [response["choices"][0]["message"]["content"] for response in responses]
    if outputs[0] != outputs[1]:
        raise RuntimeError("native greedy text changed across two captures")
    prompt_counts = [int(response["usage"]["prompt_tokens"]) for response in responses]
    if len(set(prompt_counts)) != 1 or prompt_counts[0] < 1:
        raise RuntimeError("native prompt-token count changed across two captures")
    usage = responses[0]["usage"]
    if int(usage["completion_tokens"]) != len(ids[0]):
        raise RuntimeError("native completion count differs from returned token IDs")
    return {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "prompt_tokens": prompt_counts[0],
        "max_tokens": max_tokens,
        "completion_tokens": len(ids[0]),
        "finish_reason": responses[0]["choices"][0]["finish_reason"],
        "output_sha256": hashlib.sha256(outputs[0].encode()).hexdigest(),
        "token_ids_sha256": hashlib.sha256(
            json.dumps(ids[0], separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="GLM-5.3-Flash-EXL3")
    parser.add_argument("--benchmarks", type=Path, required=True)
    parser.add_argument("--equivalence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ensure_output_path(args.output, args.force)
    base_url = args.url.rstrip("/")
    attestation_bytes = args.attestation.read_bytes()
    attestation = json.loads(attestation_bytes)
    attestation_sha256 = hashlib.sha256(attestation_bytes).hexdigest()
    health_before = get(base_url + "/health", 10)
    source_before = health_source(
        health_before, args.model, attestation, args.expected_image_id
    )
    models = get(base_url + "/v1/models", 10)
    model_records = [item for item in models.get("data", []) if item.get("id") == args.model]
    if len(model_records) != 1 or model_records[0].get("owned_by") != "tensorfold":
        raise RuntimeError(f"native TensorFold endpoint does not uniquely serve {args.model}")

    entries: dict[str, dict[str, Any]] = {}
    for spec in json.loads(args.benchmarks.read_text()):
        prompt = spec.get("prompt")
        target = spec.get("target_prompt_tokens")
        if prompt is None:
            prompt, count = sized_prompt(
                base_url,
                args.model,
                int(target),
                int(spec["seed"]),
                args.timeout,
            )
            if count != int(target):
                raise RuntimeError("sized prompt token count changed")
        value = capture(base_url, args.model, prompt, int(spec["max_tokens"]), args.timeout)
        if target is not None and value["prompt_tokens"] != int(target):
            raise RuntimeError("captured prompt-token count differs from the benchmark target")
        value["suite"] = "benchmark"
        entries[spec["id"]] = value

    for task in json.loads(args.equivalence.read_text()):
        validator, expected = validation_spec(task)
        value = capture(base_url, args.model, task["prompt"], 1024, args.timeout)
        value.update({"suite": "equivalence", "validator": validator, "expected": expected})
        entries[task["id"]] = value

    health_after = get(base_url + "/health", 10)
    if hashlib.sha256(args.attestation.read_bytes()).hexdigest() != attestation_sha256:
        raise RuntimeError("native container attestation changed during reference capture")
    source_after = health_source(
        health_after, args.model, attestation, args.expected_image_id
    )
    if source_after != source_before:
        raise RuntimeError("native source identity changed during reference capture")

    result = {
        "schema": 2,
        "captured_at_unix_ns": time.time_ns(),
        "source": {
            **source_before,
            "request_drafts": False,
            "attestation_sha256": attestation_sha256,
            "health_before": health_evidence(health_before, source_before),
            "health_after": health_evidence(health_after, source_after),
            "model_registry": model_records[0],
        },
        "benchmark_specs_sha256": hashlib.sha256(args.benchmarks.read_bytes()).hexdigest(),
        "equivalence_prompts_sha256": hashlib.sha256(args.equivalence.read_bytes()).hexdigest(),
        "entries": entries,
    }
    write_output(args.output, result, args.force)
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
