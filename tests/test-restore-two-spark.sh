#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cat >"$tmp/config.env" <<'EOF'
HEAD_SSH=test-head
WORKER_SSH=test-worker
STUDIO_SSH=test-studio
GLM_RECIPE_DIR=/srv/pinned-recipe
MCDMA_STUDIO_DIR=/srv/mcdma
GLM_PORT=8888
STUDIO_RDMA_DEVICE=test-studio-device
STUDIO_RDMA_INTERFACE=test-studio-interface
SPARK_RDMA_DEVICE=test-spark-device
SPARK_RDMA_INTERFACE=test-spark-interface
EOF

cat >"$tmp/ssh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf 'ssh-called\n' >>"$SSH_CALL_LOG"
command=${!#}
case "$command" in
  *"rev-parse HEAD"*) exit 0 ;;
  *"./stop.sh"*) exit 0 ;;
  *"DRAFTER=mtp PREPARE=0 ./start.sh"*) exit 0 ;;
  *"docker inspect 'glm53-flash-tf'"*)
    case "${MOCK_MODE:-success}" in
      oom) printf 'running|true|true|0|healthy|sha256:446a23697c7eba9e0434cb162b5389792c7a5ab0f2652173bf1e791e7fdb4c5d\n' ;;
      restarted) printf 'running|true|false|1|healthy|sha256:446a23697c7eba9e0434cb162b5389792c7a5ab0f2652173bf1e791e7fdb4c5d\n' ;;
      unhealthy) printf 'running|true|false|0|unhealthy|sha256:446a23697c7eba9e0434cb162b5389792c7a5ab0f2652173bf1e791e7fdb4c5d\n' ;;
      wrong_image) printf 'running|true|false|0|healthy|sha256:wrong\n' ;;
      *) printf 'running|true|false|0|healthy|sha256:446a23697c7eba9e0434cb162b5389792c7a5ab0f2652173bf1e791e7fdb4c5d\n' ;;
    esac
    ;;
  *"/v1/models"*)
    if [[ "${MOCK_MODE:-success}" == wrong_model ]]; then
      printf '{"data":[{"id":"wrong-model"}]}\n'
    else
      printf '{"data":[{"id":"GLM-5.3-Flash-EXL3"}]}\n'
    fi
    ;;
  *"/v1/chat/completions"*)
    if [[ "${MOCK_MODE:-success}" == wrong_answer ]]; then
      printf '{"choices":[{"finish_reason":"stop","message":{"content":"wrong"}}]}\n'
    else
      printf '{"choices":[{"finish_reason":"stop","message":{"content":"GLM_READY"}}]}\n'
    fi
    ;;
  *)
    printf 'unexpected mocked ssh command: %s\n' "$command" >&2
    exit 90
    ;;
esac
EOF
chmod +x "$tmp/ssh"

export CONFIG_FILE=$tmp/config.env
export SSH_CALL_LOG=$tmp/ssh-calls
export PATH=$tmp:$PATH

dry_output=$($root/scripts/restore-two-spark.sh --dry-run)
[[ ! -e "$SSH_CALL_LOG" ]]
grep -q '^DRY-RUN: no SSH command will be executed and no host will be changed\.$' <<<"$dry_output"
grep -q '^PLAN 1: verify the pinned recipe' <<<"$dry_output"
grep -q '^PLAN 2: stop the current experiment' <<<"$dry_output"
grep -q '^PLAN 3: start the pinned live two-Spark service' <<<"$dry_output"
grep -q '^PLAN 4: check the head container' <<<"$dry_output"
grep -q '^PLAN 5: check the worker container' <<<"$dry_output"
grep -q '^PLAN 6: require the API model ID' <<<"$dry_output"
grep -q '^PLAN 7: require a stopped-normal response' <<<"$dry_output"
! grep -q 'TWO_SPARK_BASELINE_OK' <<<"$dry_output"

: >"$SSH_CALL_LOG"
success_output=$($root/scripts/restore-two-spark.sh)
[[ $(wc -l <"$SSH_CALL_LOG" | tr -d ' ') == 7 ]]
[[ $(grep -xc 'TWO_SPARK_BASELINE_OK' <<<"$success_output") == 1 ]]

expect_failure() {
  local mode=$1 expected=$2 failure_output failure_rc
  : >"$SSH_CALL_LOG"
  set +e
  failure_output=$(MOCK_MODE=$mode $root/scripts/restore-two-spark.sh 2>&1)
  failure_rc=$?
  set -e
  [[ $failure_rc -ne 0 ]]
  grep -q "$expected" <<<"$failure_output"
  ! grep -qx 'TWO_SPARK_BASELINE_OK' <<<"$failure_output"
}

expect_failure oom 'container was OOM-killed'
expect_failure restarted 'container restart count is 1, expected 0'
expect_failure unhealthy 'container health is unhealthy'
expect_failure wrong_image 'container image is not the pinned image'
expect_failure wrong_model 'API models were'
expect_failure wrong_answer 'fixed-answer check returned'

printf 'PASS: rollback dry-run is inert and success marker is health-gated\n'
