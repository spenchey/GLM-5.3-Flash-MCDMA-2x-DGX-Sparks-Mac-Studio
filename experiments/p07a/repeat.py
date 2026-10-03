"""Require deterministic evidence from two independently exited fixture processes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

command = [sys.executable, str(Path(__file__).with_name("fixture.py")), *sys.argv[1:]]
reports = []
for _ in range(2):
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    reports.append(json.loads(completed.stdout))

stable = [{key: value for key, value in report.items() if key not in {"elapsed_seconds", "memory"}}
          for report in reports]
assert stable[0] == stable[1], "independent process output/state/identity evidence differs"
encoded = json.dumps(stable[0], sort_keys=True, separators=(",", ":")).encode()
print(json.dumps({"independent_processes": 2, "identical": True,
                  "stable_evidence_sha256": hashlib.sha256(encoded).hexdigest(),
                  "runs": [{"elapsed_seconds": r["elapsed_seconds"], "memory": r["memory"]} for r in reports],
                  "evidence": stable[0]}, indent=2, sort_keys=True))
