from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.autoresearch.evaluator import (
    ManifestError,
    content_hash,
    evaluate,
    load_json,
    validate_trial,
    write_receipt,
)


FIXTURES = Path(__file__).with_name("fixtures")


class EvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = load_json(FIXTURES / "baseline.json")

    def result(self, name: str):
        return evaluate(self.baseline, load_json(FIXTURES / name))

    def test_keep(self):
        result = self.result("keep.json")
        self.assertEqual("keep", result["verdict"])
        self.assertFalse(result["stop"])
        self.assertEqual(["window_rows"], result["changed_settings"])

    def test_wrong_answer_rejected_before_performance(self):
        result = self.result("reject-wrong-answer.json")
        self.assertEqual("reject", result["verdict"])
        self.assertEqual(["wrong answer"], result["reasons"])
        self.assertFalse(result["performance_considered"])
        self.assertIsNone(result["improvement"])

    def test_oom_rejected_and_stops(self):
        result = self.result("reject-oom.json")
        self.assertIn("OOM", result["reasons"])
        self.assertTrue(result["stop"])

    def test_service_loss_rejected_and_stops(self):
        result = self.result("reject-service-loss.json")
        self.assertIn("service loss", result["reasons"])
        self.assertTrue(result["stop"])

    def test_cleanup_failure_rejected_and_stops(self):
        result = self.result("reject-cleanup.json")
        self.assertIn("cleanup failure", result["reasons"])
        self.assertTrue(result["stop"])

    def test_recovery_failure_rejected_and_stops(self):
        result = self.result("reject-recovery.json")
        self.assertIn("recovery failure", result["reasons"])
        self.assertTrue(result["stop"])

    def test_multiple_setting_change_rejected(self):
        result = self.result("reject-multiple-settings.json")
        self.assertIn("multiple-setting change", result["reasons"])
        self.assertFalse(result["performance_considered"])

    def test_json_types_make_three_distinct_setting_changes(self):
        baseline = json.loads(json.dumps(self.baseline))
        baseline["settings"] = {"enabled": True, "fallback": False, "count": 1}
        trial = load_json(FIXTURES / "keep.json")
        trial["baseline_sha256"] = content_hash(baseline)
        trial["settings"] = {"enabled": 1, "fallback": 0, "count": 2}
        result = evaluate(baseline, trial)
        self.assertEqual(["count", "enabled", "fallback"], result["changed_settings"])
        self.assertIn("multiple-setting change", result["reasons"])

    def test_integer_to_float_is_a_setting_change(self):
        baseline = json.loads(json.dumps(self.baseline))
        trial = load_json(FIXTURES / "keep.json")
        trial["settings"] = dict(baseline["settings"])
        trial["settings"]["batch_size"] = 1.0
        trial["baseline_sha256"] = content_hash(baseline)
        self.assertEqual(["batch_size"], evaluate(baseline, trial)["changed_settings"])

    def test_performance_failure_rejected_after_correctness(self):
        result = self.result("reject-performance.json")
        self.assertEqual(["performance threshold not met"], result["reasons"])
        self.assertTrue(result["performance_considered"])

    def test_receipt_is_repeatable_and_content_addressed(self):
        first = self.result("keep.json")
        second = self.result("keep.json")
        self.assertEqual(first, second)
        with tempfile.TemporaryDirectory() as directory:
            path1 = write_receipt(Path(directory), first)
            path2 = write_receipt(Path(directory), second)
            self.assertEqual(path1, path2)
            self.assertEqual(1, len(list(Path(directory).iterdir())))
            self.assertEqual(first, json.loads(path1.read_text()))

    def test_receipt_is_strict_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_receipt(Path(directory), self.result("keep.json"))
            parsed = json.loads(
                path.read_text(),
                parse_constant=lambda value: self.fail(f"non-finite value: {value}"),
            )
            self.assertEqual("keep", parsed["verdict"])
            with self.assertRaises(ValueError):
                write_receipt(Path(directory), {"improvement": math.nan})

    def test_performance_overflow_rejects_safely(self):
        baseline = json.loads(json.dumps(self.baseline))
        baseline["evaluator"]["performance"]["baseline_value"] = 1e308
        trial = load_json(FIXTURES / "keep.json")
        trial["baseline_sha256"] = content_hash(baseline)
        trial["observations"]["performance"]["elapsed_ms"] = -1e308
        result = evaluate(baseline, trial)
        self.assertEqual("reject", result["verdict"])
        self.assertEqual(["invalid performance improvement"], result["reasons"])
        self.assertIsNone(result["improvement"])

    def test_schema_version_rejects_boolean(self):
        baseline = json.loads(json.dumps(self.baseline))
        baseline["schema_version"] = True
        with self.assertRaisesRegex(ManifestError, "schema_version"):
            validate_trial(load_json(FIXTURES / "keep.json"), baseline)

    def test_load_rejects_duplicate_keys_and_nonfinite_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(ManifestError, "duplicate JSON key"):
                load_json(duplicate)
            for value in ("NaN", "Infinity", "-Infinity"):
                invalid = Path(directory) / "invalid.json"
                invalid.write_text(f'{{"value":{value}}}')
                with self.assertRaisesRegex(ManifestError, "non-finite JSON number"):
                    load_json(invalid)

    def test_evaluation_never_opens_a_network_connection(self):
        with patch("socket.socket", side_effect=AssertionError("network attempted")):
            self.assertEqual("keep", self.result("keep.json")["verdict"])

    def test_baseline_hash_mismatch_is_invalid(self):
        trial = load_json(FIXTURES / "keep.json")
        trial["baseline_sha256"] = "0" * 64
        with self.assertRaisesRegex(ManifestError, "frozen baseline"):
            validate_trial(trial, self.baseline)


if __name__ == "__main__":
    unittest.main()
