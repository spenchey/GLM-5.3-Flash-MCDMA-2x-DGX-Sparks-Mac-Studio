#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"
require_three_machine_config

temporary=$(mktemp -d "${TMPDIR:-/tmp}/glm53-three-machine-benchmark.XXXXXXXX")
cleanup_benchmark() {
  local rc=$?
  rm -rf -- "$temporary"
  exit "$rc"
}
trap cleanup_benchmark EXIT INT TERM HUP

head_before=$(three_machine_container_state head)
worker_before=$(three_machine_container_state worker)
validate_three_machine_container_state "head Spark" "$head_before"
validate_three_machine_container_state "worker Spark" "$worker_before"
three_machine_container_running "$head_before" || die "head Spark is not running before benchmark"
three_machine_container_running "$worker_before" || die "worker Spark is not running before benchmark"
head_id=$(three_machine_container_id "$head_before")
worker_id=$(three_machine_container_id "$worker_before")
head_log_since=$(head_ssh "date -u +%Y-%m-%dT%H:%M:%S.%NZ")
worker_log_since=$(worker_ssh "date -u +%Y-%m-%dT%H:%M:%S.%NZ")
link_before=$(adapter peek)
printf '%s' "$head_before" >"$temporary/head-before.txt"
printf '%s' "$worker_before" >"$temporary/worker-before.txt"
printf '%s' "$link_before" >"$temporary/link-before.json"

api_pid=$(verified_studio_api_listener_pid)
[[ -n "$api_pid" ]] || die "the managed Mac API is not running"
rank_host_state() {
  local role=$1 container_id=$2
  if [[ "$role" == head ]]; then
    head_ssh "date -u +%Y-%m-%dT%H:%M:%SZ; awk '/MemAvailable|MemFree|Cached/ {print}' /proc/meminfo; rank_pid=\$(docker top '$container_id' -eo pid,args | awk 'NR > 1 && \$2 ~ /^python(3([.][0-9]+)*)?$/ && /[/]work[/]experiments[/]three_machine[/]cuda_partial[.]py/ {print \$1; exit}'); test -n \"\$rank_pid\" || exit 41; printf 'rank_python_pid=%s\\n' \"\$rank_pid\"; awk '/VmRSS|VmHWM/ {print}' /proc/\$rank_pid/status"
  else
    worker_ssh "date -u +%Y-%m-%dT%H:%M:%SZ; awk '/MemAvailable|MemFree|Cached/ {print}' /proc/meminfo; rank_pid=\$(docker top '$container_id' -eo pid,args | awk 'NR > 1 && \$2 ~ /^python(3([.][0-9]+)*)?$/ && /[/]work[/]experiments[/]three_machine[/]cuda_partial[.]py/ {print \$1; exit}'); test -n \"\$rank_pid\" || exit 41; printf 'rank_python_pid=%s\\n' \"\$rank_pid\"; awk '/VmRSS|VmHWM/ {print}' /proc/\$rank_pid/status"
  fi
}
head_host_before=$(rank_host_state head "$head_id")
worker_host_before=$(rank_host_state worker "$worker_id")
studio_host_before=$(studio_ssh "printf 'date_utc='; date -u +%Y-%m-%dT%H:%M:%SZ; printf 'process_rss_kib='; ps -p '$api_pid' -o rss= | tr -d ' '; pmset -g therm 2>/dev/null || true")
printf '%s' "$head_host_before" >"$temporary/head-host-before.txt"
printf '%s' "$worker_host_before" >"$temporary/worker-host-before.txt"
printf '%s' "$studio_host_before" >"$temporary/studio-host-before.txt"
head_ssh "docker inspect '$head_id' --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(HF_HUB_OFFLINE|NCCL_|TF_GLM_|THREE_MACHINE_)' | LC_ALL=C sort" >"$temporary/head-container-env.txt"
worker_ssh "docker inspect '$worker_id' --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E '^(HF_HUB_OFFLINE|NCCL_|TF_GLM_|THREE_MACHINE_)' | LC_ALL=C sort" >"$temporary/worker-container-env.txt"
shasum -a 256 "$PROJECT_ROOT/scripts/benchmark-three-machine.py" \
  "$PROJECT_ROOT/scripts/benchmark-three-machine.sh" \
  "$PROJECT_ROOT/scripts/benchmark-evidence.py" \
  "$PROJECT_ROOT/experiments/three_machine/cuda_partial.py" \
  >"$temporary/measurement-code-sha256.txt"

quoted_args=
for argument in "$@"; do
  printf -v quoted_argument '%q' "$argument"
  quoted_args+=" $quoted_argument"
done
[[ "${THREE_MACHINE_NATIVE_REFERENCE_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] \
  || die "THREE_MACHINE_NATIVE_REFERENCE_SHA256 is not a pinned SHA-256"
printf -v reference_sha256 '%q' "$THREE_MACHINE_NATIVE_REFERENCE_SHA256"
printf -v native_image_id '%q' "$THREE_MACHINE_IMAGE_ID"
benchmark=$(studio_ssh "cd '$THREE_MACHINE_STUDIO_DIR' && '$THREE_MACHINE_STUDIO_PYTHON' scripts/benchmark-three-machine.py --url 'http://$THREE_MACHINE_API_HOST:$THREE_MACHINE_API_PORT/v1/chat/completions' --reference-sha256 $reference_sha256 --expected-native-image-id $native_image_id$quoted_args")
printf '%s' "$benchmark" >"$temporary/benchmark.json"
head_ssh "docker logs --since '$head_log_since' '$head_id' 2>&1" >"$temporary/head-runtime.log"
worker_ssh "docker logs --since '$worker_log_since' '$worker_id' 2>&1" >"$temporary/worker-runtime.log"

head_after=$(three_machine_container_state head)
worker_after=$(three_machine_container_state worker)
validate_three_machine_container_state "head Spark" "$head_after"
validate_three_machine_container_state "worker Spark" "$worker_after"
three_machine_container_running "$head_after" || die "head Spark stopped during benchmark"
three_machine_container_running "$worker_after" || die "worker Spark stopped during benchmark"
[[ "$(three_machine_container_id "$head_after")" == "$(three_machine_container_id "$head_before")" ]] \
  || die "head Spark identity changed during benchmark"
[[ "$(three_machine_container_id "$worker_after")" == "$(three_machine_container_id "$worker_before")" ]] \
  || die "worker Spark identity changed during benchmark"
link_after=$(adapter peek)
printf '%s' "$head_after" >"$temporary/head-after.txt"
printf '%s' "$worker_after" >"$temporary/worker-after.txt"
printf '%s' "$link_after" >"$temporary/link-after.json"

head_host_after=$(rank_host_state head "$head_id")
worker_host_after=$(rank_host_state worker "$worker_id")
studio_host_after=$(studio_ssh "printf 'date_utc='; date -u +%Y-%m-%dT%H:%M:%SZ; printf 'process_rss_kib='; ps -p '$api_pid' -o rss= | tr -d ' '; pmset -g therm 2>/dev/null || true")
printf '%s' "$head_host_after" >"$temporary/head-host-after.txt"
printf '%s' "$worker_host_after" >"$temporary/worker-host-after.txt"
printf '%s' "$studio_host_after" >"$temporary/studio-host-after.txt"

python3 "$PROJECT_ROOT/scripts/benchmark-evidence.py" "$temporary"
