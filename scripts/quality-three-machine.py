#!/usr/bin/env python3
"""Run the bounded Mia-derived greedy code-quality/equivalence probe."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


MODEL = "GLM-5.3-Flash-TensorFold-MCDMA-3Machine"
IDENTITY = (
    "deployment_id", "commit", "project_manifest", "config_sha256", "image_id",
    "link_generation",
)


def require_reference_hash(path: Path, expected: str | None) -> str:
    if not isinstance(expected, str) or len(expected) != 64 \
            or any(char not in "0123456789abcdef" for char in expected):
        raise SystemExit("THREE_MACHINE_NATIVE_REFERENCE_SHA256 must be a full lowercase SHA-256")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit("native reference file SHA-256 differs from the pinned value")
    return observed


def require_prompt_tokens(task_id: str, observed: Any, expected: int) -> int:
    count = int(observed)
    if count != expected:
        raise RuntimeError(f"{task_id} prompt-token count differs from the native reference")
    return count


def get(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def post(url: str, prompt: str, max_tokens: int, timeout: float) -> dict[str, Any]:
    raw = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "enable_thinking": False,
        "stream": False,
    }, separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def code_body(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    parts = stripped.split("```")
    if len(parts) < 3:
        raise ValueError("response has an unclosed code fence")
    body = parts[1].strip()
    if body.startswith("python"):
        body = body[len("python"):].lstrip("\r\n ")
    return body


def validate_code(text: str, expected: list[str]) -> None:
    tree = ast.parse(code_body(text))
    names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    missing = sorted(set(expected) - names)
    if missing:
        raise ValueError(f"generated code is missing required symbol(s): {', '.join(missing)}")


def validation_spec(task: dict[str, Any]) -> tuple[str, Any]:
    """Normalize the task's response contract, including the legacy code form."""
    validator = task.get("validator")
    expected = task.get("expected")
    if validator is None and "expected_symbols" in task:
        validator = "python_symbols"
        expected = task["expected_symbols"]

    if validator == "python_symbols":
        if not isinstance(expected, list) or not expected \
                or any(not isinstance(item, str) or not item for item in expected):
            raise ValueError("python_symbols requires a non-empty string list in expected")
    elif validator == "exact_json":
        if "expected" not in task:
            raise ValueError("exact_json requires expected")
    elif validator == "integer":
        if type(expected) is not int:
            raise ValueError("integer requires an integer expected value")
    elif validator == "contains_all":
        if not isinstance(expected, list) or not expected \
                or any(not isinstance(item, str) or not item for item in expected):
            raise ValueError("contains_all requires a non-empty string list in expected")
    elif validator == "min_words":
        if type(expected) is not int or expected < 1:
            raise ValueError("min_words requires a positive integer expected value")
    else:
        raise ValueError(f"unsupported response validator: {validator!r}")
    return str(validator), expected


def fenced_body(text: str) -> str:
    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    parts = stripped.split("```")
    if len(parts) != 3 or parts[0].strip() or parts[2].strip():
        raise ValueError("response must contain exactly one complete fenced value")
    body = parts[1].strip()
    first, separator, rest = body.partition("\n")
    if separator and first.strip().lower() in {"json", "python", "text"}:
        body = rest.strip()
    return body


def validate_response(text: str, validator: str, expected: Any) -> None:
    if validator == "python_symbols":
        validate_code(text, list(expected))
        return
    if validator == "exact_json":
        try:
            actual = json.loads(fenced_body(text))
        except json.JSONDecodeError as error:
            raise ValueError("response is not exact JSON") from error
        if actual != expected:
            raise ValueError("response JSON differs from expected")
        return
    if validator == "integer":
        value = text.strip()
        if re.fullmatch(r"[+-]?\d+", value) is None or int(value) != expected:
            raise ValueError("response integer differs from expected")
        return
    if validator == "contains_all":
        missing = [item for item in expected if item not in text]
        if missing:
            raise ValueError(f"response is missing required text: {', '.join(missing)}")
        return
    if validator == "min_words":
        words = re.findall(r"\b\w+(?:['’-]\w+)*\b", text, flags=re.UNICODE)
        if len(words) < expected:
            raise ValueError(f"response has {len(words)} words, expected at least {expected}")
        return
    raise ValueError(f"unsupported response validator: {validator!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument(
        "--references",
        type=Path,
        default=Path("experiments/three_machine/native-token-references.json"),
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--reference-sha256",
        default=os.environ.get("THREE_MACHINE_NATIVE_REFERENCE_SHA256"),
    )
    parser.add_argument("--expected-native-image-id", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.max_tokens <= 2051:
        raise SystemExit("repeats must be positive and max tokens must be 1..2051")
    tasks = json.loads(args.tasks.read_text())
    if not isinstance(tasks, list) or len(tasks) != 8:
        raise SystemExit("quality probe requires the registered eight-task set")
    reference_file_sha256 = require_reference_hash(args.references, args.reference_sha256)
    reference_file = json.loads(args.references.read_text())
    source = reference_file.get("source")
    if not isinstance(source, dict) or source.get("image_id") != args.expected_native_image_id:
        raise SystemExit("native reference was not captured from the pinned TensorFold image")
    references = reference_file.get("entries") or {}
    for task in tasks:
        try:
            validator, expected = validation_spec(task)
        except ValueError as error:
            raise SystemExit(f"{task.get('id', 'task')} has an invalid response validator: {error}") from error
        reference = references.get(task.get("id"))
        if not isinstance(reference, dict) or reference.get("suite") != "equivalence":
            raise SystemExit(f"{task.get('id', 'task')} is missing its native reference")
        try:
            reference_validator, reference_expected = validation_spec(reference)
        except ValueError as error:
            raise SystemExit(f"{task['id']} native reference has an invalid validator: {error}") from error
        if reference.get("prompt") != task.get("prompt") \
                or reference_validator != validator or reference_expected != expected:
            raise SystemExit(f"{task['id']} native reference describes a different task")
        if int(reference.get("max_tokens", 0)) != args.max_tokens:
            raise SystemExit(f"{task['id']} max_tokens differs from its native reference")
        if type(reference.get("prompt_tokens")) is not int or reference["prompt_tokens"] < 1:
            raise SystemExit(f"{task['id']} native reference has no valid prompt_tokens")
        if hashlib.sha256(task["prompt"].encode()).hexdigest() != reference.get("prompt_sha256"):
            raise SystemExit(f"{task['id']} native prompt hash is invalid")
        token_hash = reference.get("token_ids_sha256")
        if not isinstance(token_hash, str) or len(token_hash) != 64 \
                or any(char not in "0123456789abcdef" for char in token_hash):
            raise SystemExit(f"{task['id']} has no valid native token reference")
    health_url = args.url.removesuffix("/v1/chat/completions") + "/health"
    before = get(health_url, 10)
    if before.get("status") != "ready" or int(before.get("failed_requests", -1)) != 0:
        raise RuntimeError("service is not clean and ready before quality probe")
    for field in IDENTITY:
        if before.get(field) in (None, "", "unknown"):
            raise RuntimeError(f"health is missing {field}")

    results = []
    expected_calls = 0
    for task in tasks:
        reference = references[task["id"]]
        validator, expected = validation_spec(task)
        task_hashes = []
        token_hashes = []
        completions = []
        for _ in range(args.repeats):
            began = time.perf_counter()
            response = post(args.url, task["prompt"], args.max_tokens, args.timeout)
            wall_s = time.perf_counter() - began
            choice = response["choices"][0]
            if choice["finish_reason"] == "length":
                raise RuntimeError(f"{task['id']} exhausted its token allowance")
            text = choice["message"]["content"]
            validate_response(text, validator, expected)
            metrics = response["three_machine_metrics"]
            token_ids = [int(value) for value in metrics["generated_token_ids"]]
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            token_hash = hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()
            ).hexdigest()
            task_hashes.append(text_hash)
            token_hashes.append(token_hash)
            usage = response["usage"]
            prompt_count = require_prompt_tokens(
                task["id"], usage["prompt_tokens"], reference["prompt_tokens"]
            )
            steps = metrics["steps"]
            prompt_steps = [step for step in steps if step["prompt"]]
            decode_steps = [step for step in steps if not step["prompt"]]
            expected_prompt_steps = math.ceil(prompt_count / 63)
            decode_rounds = int(metrics.get("decode_rounds", -1))
            expected_steps = expected_prompt_steps + decode_rounds
            if len(prompt_steps) != expected_prompt_steps \
                    or len(decode_steps) != decode_rounds \
                    or len(steps) != expected_steps:
                raise RuntimeError(f"{task['id']} returned the wrong number of protocol steps")
            if any(not 1 <= int(step.get("rows", 0)) <= 63 for step in steps):
                raise RuntimeError(f"{task['id']} returned an invalid MCDMA window size")
            if sum(int(step.get("accepted_tokens", 0)) for step in steps) \
                    != int(usage["completion_tokens"]):
                raise RuntimeError(f"{task['id']} returned inconsistent accepted-token evidence")
            if float(metrics.get("metal_s", 0)) <= 0 \
                    or any(float(step.get("metal_s", 0)) <= 0 for step in steps):
                raise RuntimeError(f"{task['id']} has no Mac Metal participation proof")
            if token_hash != reference["token_ids_sha256"]:
                raise RuntimeError(f"{task['id']} token IDs differ from the native reference")
            expected_calls += 2 + 2 * expected_steps
            completions.append({
                "wall_s": wall_s,
                "request_id": metrics["request_id"],
                "generation": metrics["generation"],
                "prompt_tokens": prompt_count,
                "reference_prompt_tokens": reference["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "output_sha256": text_hash,
                "token_ids_sha256": token_hash,
                "reference_token_ids_sha256": reference["token_ids_sha256"],
                "first_token_s": metrics["first_token_s"],
                "steps": [
                    {"step": int(step["step"]), "prompt": bool(step["prompt"])}
                    for step in steps
                ],
            })
        if len(set(task_hashes)) != 1 or len(set(token_hashes)) != 1:
            raise RuntimeError(f"{task['id']} changed across greedy repeats")
        results.append({
            "id": task["id"],
            "prompt_sha256": hashlib.sha256(task["prompt"].encode()).hexdigest(),
            "validator": validator,
            "expected": expected,
            "completions": completions,
        })

    after = get(health_url, 10)
    for field in IDENTITY:
        if after.get(field) != before.get(field):
            raise RuntimeError(f"{field} changed during quality probe")
    if after.get("status") != "ready" or int(after.get("failed_requests", -1)) != 0:
        raise RuntimeError("service failed during quality probe")
    if int(after["completed_requests"]) - int(before["completed_requests"]) != len(tasks) * args.repeats:
        raise RuntimeError("completed request count differs from quality probe request count")
    print(json.dumps({
        "schema": 1,
        "captured_at_unix_ns": time.time_ns(),
        "task_set_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "reference_file_sha256": reference_file_sha256,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "expected_mcdma_calls": expected_calls,
        "health_before": before,
        "health_after": after,
        "results": results,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
