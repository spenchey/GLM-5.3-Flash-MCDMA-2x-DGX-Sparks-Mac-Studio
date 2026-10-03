"""Validate and evaluate one-variable experiments without network access."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
FAILURE_KEYS = ("oom", "service_loss", "cleanup_failure", "recovery_failure")


class ManifestError(ValueError):
    """The baseline or trial does not meet the frozen contract."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ManifestError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ManifestError(f"non-finite JSON number: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain one JSON object")
    return value


def _require_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ManifestError(f"{label} keys must be exactly {sorted(expected)}; got {sorted(actual)}")


def validate_baseline(baseline: dict[str, Any]) -> None:
    _require_keys(baseline, {"schema_version", "baseline_id", "settings", "evaluator"}, "baseline")
    if type(baseline["schema_version"]) is not int or baseline["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(baseline["baseline_id"], str) or not baseline["baseline_id"]:
        raise ManifestError("baseline_id must be a non-empty string")
    if not isinstance(baseline["settings"], dict) or not baseline["settings"]:
        raise ManifestError("settings must be a non-empty object")
    evaluator = baseline["evaluator"]
    if not isinstance(evaluator, dict):
        raise ManifestError("evaluator must be an object")
    _require_keys(evaluator, {"correctness_checks", "performance"}, "evaluator")
    checks = evaluator["correctness_checks"]
    if not isinstance(checks, list) or not checks or any(not isinstance(x, str) or not x for x in checks):
        raise ManifestError("correctness_checks must be a non-empty list of names")
    if len(set(checks)) != len(checks):
        raise ManifestError("correctness_checks must not contain duplicates")
    performance = evaluator["performance"]
    if not isinstance(performance, dict):
        raise ManifestError("performance must be an object")
    _require_keys(performance, {"metric", "direction", "baseline_value", "minimum_improvement"}, "performance")
    if not isinstance(performance["metric"], str) or not performance["metric"]:
        raise ManifestError("performance metric must be a non-empty string")
    if performance["direction"] not in {"higher", "lower"}:
        raise ManifestError("performance direction must be higher or lower")
    for key in ("baseline_value", "minimum_improvement"):
        if isinstance(performance[key], bool) or not isinstance(performance[key], (int, float)):
            raise ManifestError(f"{key} must be a number")
        try:
            finite = math.isfinite(performance[key])
        except OverflowError:
            finite = False
        if not finite:
            raise ManifestError(f"{key} must be finite")
    if performance["minimum_improvement"] < 0:
        raise ManifestError("minimum_improvement must be zero or greater")


def validate_trial(trial: dict[str, Any], baseline: dict[str, Any]) -> None:
    validate_baseline(baseline)
    _require_keys(trial, {"schema_version", "trial_id", "baseline_sha256", "settings", "observations"}, "trial")
    if type(trial["schema_version"]) is not int or trial["schema_version"] != SCHEMA_VERSION:
        raise ManifestError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(trial["trial_id"], str) or not trial["trial_id"]:
        raise ManifestError("trial_id must be a non-empty string")
    if trial["baseline_sha256"] != content_hash(baseline):
        raise ManifestError("baseline_sha256 does not match the frozen baseline")
    if not isinstance(trial["settings"], dict):
        raise ManifestError("trial settings must be an object")
    observations = trial["observations"]
    if not isinstance(observations, dict):
        raise ManifestError("observations must be an object")
    _require_keys(observations, {"correctness", "failures", "performance"}, "observations")
    correctness = observations["correctness"]
    if not isinstance(correctness, dict) or set(correctness) != set(baseline["evaluator"]["correctness_checks"]):
        raise ManifestError("correctness results must exactly match the frozen check names")
    if any(not isinstance(value, bool) for value in correctness.values()):
        raise ManifestError("correctness results must be true or false")
    failures = observations["failures"]
    if not isinstance(failures, dict) or set(failures) != set(FAILURE_KEYS):
        raise ManifestError(f"failure results must exactly contain {list(FAILURE_KEYS)}")
    if any(not isinstance(value, bool) for value in failures.values()):
        raise ManifestError("failure results must be true or false")
    performance = observations["performance"]
    metric = baseline["evaluator"]["performance"]["metric"]
    if not isinstance(performance, dict) or set(performance) != {metric}:
        raise ManifestError(f"performance must contain only the frozen metric {metric}")
    value = performance[metric]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError("performance result must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ManifestError("performance result must be finite")


def _changed_settings(before: Any, after: Any, prefix: str = "") -> list[str]:
    if type(before) is not type(after):
        return [prefix]
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[str] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                changes.append(path)
            else:
                changes.extend(_changed_settings(before[key], after[key], path))
        return changes
    if isinstance(before, list) and isinstance(after, list):
        changes = []
        for index in range(max(len(before), len(after))):
            path = f"{prefix}[{index}]"
            if index >= len(before) or index >= len(after):
                changes.append(path)
            else:
                changes.extend(_changed_settings(before[index], after[index], path))
        return changes
    return [] if before == after else [prefix]


def evaluate(baseline: dict[str, Any], trial: dict[str, Any]) -> dict[str, Any]:
    validate_trial(trial, baseline)
    changes = _changed_settings(baseline["settings"], trial["settings"])
    reasons: list[str] = []
    if len(changes) != 1:
        reasons.append("multiple-setting change" if len(changes) > 1 else "no setting changed")

    correctness = trial["observations"]["correctness"]
    if not all(correctness.values()):
        reasons.append("wrong answer")
    failures = trial["observations"]["failures"]
    labels = {
        "oom": "OOM",
        "service_loss": "service loss",
        "cleanup_failure": "cleanup failure",
        "recovery_failure": "recovery failure",
    }
    reasons.extend(labels[key] for key in FAILURE_KEYS if failures[key])

    # Correctness and safety always decide before performance is considered.
    performance_considered = not reasons
    improvement: float | None = None
    if performance_considered:
        contract = baseline["evaluator"]["performance"]
        observed = trial["observations"]["performance"][contract["metric"]]
        if contract["direction"] == "higher":
            improvement = observed - contract["baseline_value"]
        else:
            improvement = contract["baseline_value"] - observed
        try:
            finite_improvement = math.isfinite(improvement)
        except OverflowError:
            finite_improvement = False
        if not finite_improvement:
            improvement = None
            reasons.append("invalid performance improvement")
        elif improvement < contract["minimum_improvement"]:
            reasons.append("performance threshold not met")

    verdict = "keep" if not reasons else "reject"
    stop = any(failures.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": baseline["baseline_id"],
        "baseline_sha256": content_hash(baseline),
        "trial_id": trial["trial_id"],
        "trial_sha256": content_hash(trial),
        "changed_settings": changes,
        "correctness_checked_before_performance": True,
        "performance_considered": performance_considered,
        "improvement": improvement,
        "verdict": verdict,
        "stop": stop,
        "reasons": reasons,
    }


def write_receipt(receipt_dir: Path, result: dict[str, Any]) -> Path:
    payload = _canonical(result)
    digest = hashlib.sha256(payload).hexdigest()
    path = receipt_dir / f"{digest}.json"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"content-address collision at {path}")
        return path
    try:
        with path.open("xb") as handle:
            handle.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise RuntimeError(f"content-address collision at {path}")
    return path
