#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_cache_handoff_config

parallel=${1:?usage: benchmark-native-allocation.sh PARALLEL [TOKENS]}
tokens=${2:-256}
[[ "$parallel" =~ ^([1-9]|1[0-6])$ ]] || die "parallel must be between 1 and 16"
[[ "$tokens" =~ ^[1-9][0-9]*$ ]] || die "token count must be positive"

prompt='Output the integers from 1 through 2000 in order, separated only by commas. Do not explain and do not stop early.'
run_id="native-capacity-p${parallel}-t${tokens}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
evidence="$PROJECT_ROOT/results/raw/$run_id"
umask 077
mkdir -p "$evidence"
native_started=false
cleanup() {
  local rc=$?
  if [[ "$native_started" == true ]]; then
    "$PROJECT_ROOT/scripts/stop-native-portable.sh" \
      >"$evidence/native-stop.stdout" 2>"$evidence/native-stop.stderr" || rc=$?
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

"$PROJECT_ROOT/scripts/start-native-portable.sh" \
  >"$evidence/native-start.stdout" 2>"$evidence/native-start.stderr"
native_started=true

head_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/head-before.json"
worker_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/worker-before.json"
native_url="http://$THREE_MACHINE_STUDIO_TO_HEAD_SSH:$THREE_MACHINE_NATIVE_PORTABLE_PORT"
studio_ssh "curl -fsS --max-time 10 '$native_url/health'" >"$evidence/studio-health-before.json"

command=(
  "$THREE_MACHINE_STUDIO_PYTHON" "$THREE_MACHINE_STUDIO_DIR/scripts/benchmark-native-portable.py"
  --url "$native_url" --model GLM-5.3-Flash-Portable --prompt "$prompt"
  --parallel "$parallel" --max-tokens "$tokens" --warmups 1 --repeats 4 --timeout 900
)
printf -v command_text '%q ' "${command[@]}"
studio_ssh "$command_text" >"$evidence/result.json" 2>"$evidence/benchmark.stderr"

head_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/head-after.json"
worker_ssh "docker inspect '$THREE_MACHINE_NATIVE_PORTABLE_CONTAINER'" >"$evidence/worker-after.json"

python3 - "$evidence/result.json" "$parallel" "$tokens" >"$evidence/acceptance.json" <<'PY'
import json, sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text())
parallel, tokens = int(sys.argv[2]), int(sys.argv[3])
if result.get("kind") != "portable-native-two-spark-concurrency":
    raise SystemExit("native benchmark returned the wrong evidence kind")
if result.get("parallel") != parallel or result.get("max_tokens") != tokens:
    raise SystemExit("native benchmark used the wrong workload")
if len(result.get("warmups") or []) != 1 or len(result.get("runs") or []) != 4:
    raise SystemExit("native benchmark did not run one warmup and four measurements")
rounds = list(result["warmups"]) + list(result["runs"])
if any(item.get("completion_tokens") != parallel * tokens for item in rounds):
    raise SystemExit("native benchmark did not complete every fixed reply")
if any(item.get("token_sha256") != result.get("token_sha256") for item in rounds):
    raise SystemExit("native output changed across rounds")
record = {
    "claim_scope": "diagnostic-only-unpaired",
    "accepted_for_three_machine_comparison": False,
    "parallel": parallel,
    "tokens": tokens,
    "token_sha256": result["token_sha256"],
    "median_tokens_per_second": result["aggregate_pipeline_tokens_per_second"]["median"],
}
print(json.dumps(record, sort_keys=True))
PY

"$PROJECT_ROOT/scripts/stop-native-portable.sh" \
  >"$evidence/native-stop.stdout" 2>"$evidence/native-stop.stderr"
native_started=false
trap - EXIT INT TERM HUP
note "PASS: unpaired two-Spark diagnostic is at $evidence; use benchmark-concurrent-capacity.sh for acceptance"
cat "$evidence/acceptance.json"
