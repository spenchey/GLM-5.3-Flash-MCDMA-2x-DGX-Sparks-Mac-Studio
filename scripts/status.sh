#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/lib.sh"

note "GLM head container"
head_ssh "docker inspect glm53-flash-tf --format 'running={{.State.Running}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}' 2>/dev/null || echo running=false"
note "GLM worker container"
head_ssh "ssh -o BatchMode=yes '$WORKER_SSH' \"docker inspect glm53-flash-tf --format='running={{.State.Running}} oom={{.State.OOMKilled}} restarts={{.RestartCount}}'\" 2>/dev/null || echo running=false"
note "GLM API"
head_ssh "curl -fsS --max-time 10 http://127.0.0.1:'$GLM_PORT'/v1/models || true"
printf '\n'
note "Mac Studio MCDMA port"
studio_ssh "ifconfig '$STUDIO_RDMA_INTERFACE' | awk '/flags=|ether |status:/'"
note "Integration"
printf 'tensorfold_mcdma_backend=not_implemented\n'
