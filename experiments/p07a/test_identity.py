"""Offline payload-range and evidence tests; no model/runtime dependency."""
import hashlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from identity import expected, hash_payload, payload_evidence


class RangeReader(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.ranges = []

    def read(self, count=-1):
        start = self.tell()
        result = super().read(count)
        self.ranges.append((start, start + len(result)))
        return result


class PayloadTests(unittest.TestCase):
    def test_exact_selected_range(self):
        selected = b"a" * (1024 * 1024 + 7)
        handle = RangeReader(b"forbidden-prefix" + selected + b"forbidden-suffix")
        self.assertEqual(hash_payload(handle, 16, len(selected)), hashlib.sha256(selected).hexdigest())
        self.assertEqual(handle.ranges, [(16, 16 + 1024 * 1024), (16 + 1024 * 1024, 16 + len(selected))])

    def test_truncation_and_bad_ranges_fail(self):
        with self.assertRaises(ValueError):
            hash_payload(io.BytesIO(b"abc"), 1, 5)
        with self.assertRaises(AssertionError):
            hash_payload(io.BytesIO(b"abc"), -1, 1)

    def test_all_53_hashes_exclude_unselected_gaps(self):
        # Tiny synthetic stored payloads use the real 53 names. Tests the hashing
        # machinery without allocating the real checkpoint's 500 MiB of tensors.
        names = sorted(expected())
        payload = bytearray(b"HEADER")
        ranges, expected_hashes, expected_reads = {}, {}, []
        for i, name in enumerate(names):
            payload.extend(b"UNSELECTED")
            start = len(payload)
            data = bytes([i]) * (i + 1)
            payload.extend(data)
            ranges[name] = ("fixture.safetensors", start, len(data))
            expected_reads.append((start, start + len(data)))
            expected_hashes[name] = hashlib.sha256(data).hexdigest()
        payload.extend(b"UNSELECTED-TAIL")
        handle = RangeReader(bytes(payload))
        with patch.object(Path, "open", return_value=handle):
            evidence = payload_evidence(Path("unused"), ranges)
        self.assertEqual(handle.ranges, expected_reads)
        self.assertEqual(evidence["sha256"], expected_hashes)
        canonical = json.dumps(expected_hashes, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(evidence["combined_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(evidence["stored_bytes_hashed"], sum(range(1, 54)))


if __name__ == "__main__":
    unittest.main()
