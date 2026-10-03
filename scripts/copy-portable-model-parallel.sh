#!/usr/bin/env bash
set -euo pipefail

# Run this on the Mac Studio.  Independent checkpoint shards are copied in
# parallel because a single encrypted Wi-Fi stream leaves most of the link
# idle.  Every rsync remains resumable and writes only its own shard.

SOURCE_DIR=${SOURCE_DIR:?set SOURCE_DIR to the resolved portable checkpoint}
DEST_HOST=${DEST_HOST:?set DEST_HOST to the destination Spark}
DEST_DIR=${DEST_DIR:?set DEST_DIR to the Spark checkpoint directory}
PARALLEL=${PARALLEL:-8}
START_SHARD=${START_SHARD:-1}
END_SHARD=${END_SHARD:-43}
STATE_DIR=${STATE_DIR:-/tmp/glm53-portable-copy}

[[ "$PARALLEL" =~ ^[1-9][0-9]*$ ]] || { printf 'PARALLEL must be positive\n' >&2; exit 2; }
[[ "$START_SHARD" =~ ^[0-9]+$ && "$END_SHARD" =~ ^[0-9]+$ ]] \
  || { printf 'shard bounds must be whole numbers\n' >&2; exit 2; }
(( START_SHARD >= 1 && END_SHARD <= 43 && START_SHARD <= END_SHARD )) \
  || { printf 'shard bounds must be inside 1..43\n' >&2; exit 2; }
[[ -d "$SOURCE_DIR" ]] || { printf 'source checkpoint does not exist\n' >&2; exit 2; }

umask 077
mkdir -p "$STATE_DIR"
ssh -o BatchMode=yes "$DEST_HOST" "mkdir -p '$DEST_DIR'"

copy_shard() {
  local shard=$1 file log
  printf -v file 'model-%05d-of-00043.safetensors' "$shard"
  log="$STATE_DIR/$file.log"
  [[ -f "$SOURCE_DIR/$file" ]] || { printf 'missing %s\n' "$file" >&2; return 2; }
  printf 'start %s %s\n' "$(date -u +%FT%TZ)" "$file" >"$log"
  rsync -aL --partial --progress "$SOURCE_DIR/$file" "$DEST_HOST:$DEST_DIR/$file" >>"$log" 2>&1
  printf 'done %s %s\n' "$(date -u +%FT%TZ)" "$file" >>"$log"
}

failures=0
active=0
pids=()
names=()
for ((shard=START_SHARD; shard<=END_SHARD; shard++)); do
  copy_shard "$shard" &
  pids+=("$!")
  names+=("$shard")
  active=$((active + 1))
  if (( active == PARALLEL || shard == END_SHARD )); then
    for index in "${!pids[@]}"; do
      if ! wait "${pids[$index]}"; then
        printf 'shard %s failed; see %s\n' "${names[$index]}" "$STATE_DIR" >&2
        failures=$((failures + 1))
      fi
    done
    pids=()
    names=()
    active=0
  fi
done
(( failures == 0 )) || exit 1

# Copy tokenizer/config/index files after the large independent shards.
rsync -aL --partial --exclude 'model-*-of-00043.safetensors' \
  "$SOURCE_DIR/" "$DEST_HOST:$DEST_DIR/" >"$STATE_DIR/metadata.log" 2>&1
printf 'complete %s\n' "$(date -u +%FT%TZ)" >"$STATE_DIR/COMPLETE"
