#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=scripts/lib.sh
source "$root/scripts/lib.sh"

stamp=$(date -u +%Y%m%dT%H%M%SZ)
raw_dir=${RESULTS_RAW_DIR:-$root/results/raw/P01-$stamp}
mkdir -p "$raw_dir"

port=${BENCHMARK_LOCAL_PORT:-18888}
tunnel_pid=''
monitor_pid=''
cleanup() {
  [[ -z "$monitor_pid" ]] || kill "$monitor_pid" 2>/dev/null || true
  [[ -z "$tunnel_pid" ]] || kill "$tunnel_pid" 2>/dev/null || true
  [[ -z "$monitor_pid" ]] || wait "$monitor_pid" 2>/dev/null || true
  [[ -z "$tunnel_pid" ]] || wait "$tunnel_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

note "Capturing read-only host-memory samples from both Sparks"
ssh -o BatchMode=yes "$HEAD_SSH" 'bash -s' -- "$WORKER_SSH" >"$raw_dir/memory-samples.csv" <<'REMOTE' &
worker=$1
printf 'epoch_ns,head_used_kib,worker_used_kib\n'
while :; do
  now=$(date +%s%N)
  head_used=$(awk '/MemTotal:/{t=$2}/MemAvailable:/{a=$2}END{print t-a}' /proc/meminfo)
  worker_used=$(ssh -o BatchMode=yes "$worker" \
    "awk '/MemTotal:/{t=\$2}/MemAvailable:/{a=\$2}END{print t-a}' /proc/meminfo")
  printf '%s,%s,%s\n' "$now" "$head_used" "$worker_used"
  sleep 0.25
done
REMOTE
monitor_pid=$!

note "Opening a temporary local tunnel to the unchanged API"
ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:$port:127.0.0.1:$GLM_PORT" "$HEAD_SSH" &
tunnel_pid=$!

for _ in $(seq 1 40); do
  curl -fsS --max-time 2 "http://127.0.0.1:$port/v1/models" >"$raw_dir/models-before.json" && break
  sleep 0.25
done
[[ -s "$raw_dir/models-before.json" ]] || die "API tunnel did not become ready"

head_ssh "docker inspect glm53-flash-tf --format 'running={{.State.Running}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}'" >"$raw_dir/head-health-before.txt"
head_ssh "ssh -o BatchMode=yes '$WORKER_SSH' \"docker inspect glm53-flash-tf --format='running={{.State.Running}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}'\"" >"$raw_dir/worker-health-before.txt"

python3 - "$port" "$raw_dir" <<'PY'
import hashlib, http.client, json, math, pathlib, statistics, sys, time

port, raw_dir = int(sys.argv[1]), pathlib.Path(sys.argv[2])
model = "GLM-5.3-Flash-EXL3"
seeds = [1701, 1702, 1703, 1704, 1705]
prompts = [
    ("exact", "Reply with exactly this text and nothing else: BASELINE_FIXED_OK"),
    ("reason", "A box has 7 rows of 8 bolts. Reply with only the total number of bolts."),
    ("transform", "Convert the lowercase text 'tensor fold baseline' to uppercase. Reply with only the converted text."),
]
results = []

def api_get(path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    started = time.perf_counter()
    conn.request("GET", path)
    response = conn.getresponse()
    body = response.read()
    elapsed = time.perf_counter() - started
    conn.close()
    return response.status, body, elapsed

for seed in seeds:
    for prompt_id, prompt in prompts:
        run_id = f"{prompt_id}-seed-{seed}"
        pre_status, pre_body, pre_health_s = api_get("/v1/models")
        if pre_status != 200:
            raise SystemExit(f"{run_id}: pre-run API health failed: HTTP {pre_status}")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "seed": seed,
            "temperature": 0,
            "max_tokens": 128,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        body = json.dumps(payload, separators=(",", ":")).encode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        request_start_epoch_ns = time.time_ns()
        request_start = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            error = response.read()
            raise SystemExit(f"{run_id}: request failed: HTTP {response.status}: {error[:500]!r}")
        raw = bytearray()
        token_events = []
        visible_parts, reasoning_parts = [], []
        usage = {}
        finish_reason = None
        while True:
            line = response.readline()
            if not line:
                break
            raw.extend(line)
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            item = json.loads(data)
            if item.get("usage"):
                usage = item["usage"]
            choices = item.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            emitted = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if emitted:
                token_events.append(time.perf_counter())
            if delta.get("content"):
                visible_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            finish_reason = choice.get("finish_reason") or finish_reason
        request_end = time.perf_counter()
        request_end_epoch_ns = time.time_ns()
        conn.close()
        raw_path = raw_dir / f"{run_id}.sse"
        raw_path.write_bytes(raw)
        post_status, post_body, post_health_s = api_get("/v1/models")
        if post_status != 200:
            raise SystemExit(f"{run_id}: post-run API health failed: HTTP {post_status}")
        ttft_s = token_events[0] - request_start if token_events else None
        intervals = [b-a for a, b in zip(token_events, token_events[1:])]
        completion_tokens = usage.get("completion_tokens")
        decode_s = token_events[-1] - token_events[0] if len(token_events) > 1 else 0
        output_rate = ((completion_tokens - 1) / decode_s
                       if completion_tokens and completion_tokens > 1 and decode_s > 0 else None)
        results.append({
            "run_id": run_id, "seed": seed, "prompt_id": prompt_id,
            "request_start_epoch_ns": request_start_epoch_ns,
            "request_end_epoch_ns": request_end_epoch_ns,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "visible_text": "".join(visible_parts),
            "visible_sha256": hashlib.sha256("".join(visible_parts).encode()).hexdigest(),
            "reasoning_sha256": hashlib.sha256("".join(reasoning_parts).encode()).hexdigest(),
            "ttft_s": ttft_s,
            "inter_token_mean_ms": statistics.mean(intervals)*1000 if intervals else None,
            "inter_token_median_ms": statistics.median(intervals)*1000 if intervals else None,
            "effective_inter_token_ms": 1000/output_rate if output_rate else None,
            "output_tokens_per_s": output_rate,
            "end_to_end_s": request_end-request_start,
            "stream_events": len(token_events), "usage": usage,
            "finish_reason": finish_reason,
            "pre_api_http": pre_status, "post_api_http": post_status,
            "pre_health_s": pre_health_s, "post_health_s": post_health_s,
        })
        print(json.dumps(results[-1]), flush=True)

# Apply a robust slow-run rule to each prompt independently so normal prompt
# complexity is not mistaken for migration. The two-second floor is over five
# times the largest ordinary TTFT observed in the initial dry measurement.
for prompt_id, _ in prompts:
    group = [r for r in results if r["prompt_id"] == prompt_id]
    values = [r["ttft_s"] for r in group if r["ttft_s"] is not None]
    median = statistics.median(values)
    mad = statistics.median(abs(x-median) for x in values)
    threshold = max(2.0*median, median + 6.0*mad, 2.0)
    for r in group:
        r["page_migration_class"] = (
            "suspected-page-migration-slow" if r["ttft_s"] is not None and r["ttft_s"] > threshold
            else "normal"
        )
        r["page_migration_ttft_threshold_s"] = threshold

(raw_dir / "metrics.json").write_text(json.dumps({
    "model": model, "seeds": seeds,
    "prompts": [{"id": p[0], "text": p[1]} for p in prompts],
    "runs": results,
}, indent=2) + "\n")
PY

curl -fsS --max-time 10 "http://127.0.0.1:$port/v1/models" >"$raw_dir/models-after.json"
head_ssh "docker inspect glm53-flash-tf --format 'running={{.State.Running}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}'" >"$raw_dir/head-health-after.txt"
head_ssh "ssh -o BatchMode=yes '$WORKER_SSH' \"docker inspect glm53-flash-tf --format='running={{.State.Running}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}'\"" >"$raw_dir/worker-health-after.txt"

cleanup
trap - EXIT INT TERM

python3 - "$raw_dir" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
data = json.loads((p / "metrics.json").read_text())
samples = []
with (p / "memory-samples.csv").open() as f:
    next(f)
    for line in f:
        try:
            ns, head, worker = map(int, line.strip().split(','))
            samples.append((ns, head, worker))
        except ValueError:
            pass
for r in data["runs"]:
    # Include the nearest surrounding samples because very short answers can
    # complete between the quarter-second samples from the remote worker.
    within = [s for s in samples if r["request_start_epoch_ns"]-500_000_000 <= s[0] <= r["request_end_epoch_ns"]+500_000_000]
    r["head_host_used_high_water_kib"] = max((s[1] for s in within), default=None)
    r["worker_host_used_high_water_kib"] = max((s[2] for s in within), default=None)
(p / "metrics.json").write_text(json.dumps(data, indent=2) + "\n")
PY

(cd "$raw_dir" && shasum -a 256 ./* > SHA256SUMS)
note "PASS: baseline evidence captured at $raw_dir"
