# P03 rollback rehearsal preflight

Date: 2026-10-01

Result: **READY FOR OWNER APPROVAL; LIVE CYCLES NOT RUN**

This is a read-only preflight for P03. No SSH command was executed. No
container, API route, MCDMA service, interface, or host was changed. The three
required stop/start cycles remain blocked on an owner-approved maintenance
window.

## Controller and test proof

The rehearsal command is the P02 controller:

```bash
./scripts/restore-two-spark.sh
```

The controller and focused test are byte-for-byte unchanged between the P02
implementation commit `2ec88f2fd2f8ad904dd4267208d8c6510590203a` and the
preflight base `3827e6fc6028371c1e7f853bd9c27b1397e4e4a6`.

- controller SHA-256:
  `3996b6a541d782265f0792cf21817a8f7935787a1c035690e8167b8896b58dba`
- focused-test SHA-256:
  `8a2dd554aaa4eef66179fca9763540dc9c9ec4da3bc37a7f11b525cec149fb45`
- dry-run output SHA-256:
  `91edcf5f22fde28059f7dd95240f93ce8cb81159e6151c4d142982b0ccb11035`

Read-only checks passed:

```text
PASS: pinned inputs and private-data guard
PASS: rollback dry-run is inert and success marker is health-gated
DRY-RUN: no SSH command will be executed and no host will be changed.
DRY-RUN COMPLETE: the success marker is intentionally withheld.
```

The focused test proves the dry-run makes no SSH call. Its mock success path
requires the marker exactly once, and its injected OOM, restart, unhealthy,
wrong-image, wrong-model, and wrong-answer cases all fail without the marker.

## Current service-state boundary

The latest retained live proof is P01 on 2026-10-01: both Spark containers were
running, not OOM-killed, at restart count zero, and the API exposed
`GLM-5.3-Flash-EXL3`; a later P04B receipt also says the live API and containers
were left unchanged. Those are historical observations, not a fresh live
check. This preflight was forbidden from touching hosts, and there is no local
listener on TCP port 8888, so current remote service health cannot be proved
until the approved window begins. The controller's first remote step rechecks
all immutable pins before it stops anything.

## Exact three-cycle sequence

Run from the project root only after approval:

```bash
set -euo pipefail
mkdir -p results/raw/P03-rollback-rehearsal
for cycle in 1 2 3; do
  date -u '+cycle='"$cycle"' start=%Y-%m-%dT%H:%M:%SZ' |
    tee "results/raw/P03-rollback-rehearsal/cycle-${cycle}.log"
  /usr/bin/time -p ./scripts/restore-two-spark.sh 2>&1 |
    tee -a "results/raw/P03-rollback-rehearsal/cycle-${cycle}.log"
  test "$(grep -xc 'TWO_SPARK_BASELINE_OK' "results/raw/P03-rollback-rehearsal/cycle-${cycle}.log")" -eq 1
  date -u '+cycle='"$cycle"' end=%Y-%m-%dT%H:%M:%SZ' |
    tee -a "results/raw/P03-rollback-rehearsal/cycle-${cycle}.log"
done
shasum -a 256 results/raw/P03-rollback-rehearsal/cycle-*.log
```

Each controller invocation performs, in order: verify the pinned clean recipe,
image, model revision, worker, and MTP settings on both Sparks; run the pinned
recipe's `./stop.sh`; run `DRAFTER=mtp PREPARE=0 ./start.sh`; inspect both
containers; require the single exact API model ID; and require one normal-stop
answer whose visible text is exactly `GLM_READY`. Only then does it print
`TWO_SPARK_BASELINE_OK`.

## Abort conditions

Stop immediately and do not begin another cycle if any command exits nonzero,
the success marker is missing or duplicated, either container is not running,
is unhealthy, was OOM-killed, restarted, or uses the wrong image, the model ID
differs, or the fixed answer differs or does not finish normally. A failed
restore is an incident: preserve its log and restore service manually through
the pinned recipe before doing any further project work.

Do not bypass the pin checks, change routing, replace a container by hand,
start MCDMA, or proceed to an experimental workload during P03.

## Expected interruption

**Inference, not measured:** reserve 5-10 minutes of service interruption per
cycle, or 15-30 minutes across all three cycles. The outage begins when
`./stop.sh` removes the serving containers and ends only after `./start.sh`
loads the model and the controller completes both API checks. The observed
controller gives the model-list check a 30-second timeout and the fixed-answer
check a 180-second timeout, but it places no upper bound on model startup, so
the estimate is a maintenance-window allowance rather than a guaranteed
maximum. Record `time -p` output in every cycle to replace this inference with
measured downtime.

## Proof and cleanup

Retain one log and SHA-256 per cycle, including UTC start/end, controller
output, timing, exactly one success marker, both container inspections, exact
model identity, and the fixed-answer result. After cycle three, make one final
read-only status capture and record both containers and the API as healthy.

This preflight created no tunnel, sampler, daemon, or long-running process. The
isolated test removed its temporary directory through its exit trap. Local
checks found no TCP 8888 listener and no surviving rollback, MCDMA daemon,
control-tunnel, or related SSH-forward process. Do not remove unrelated
untracked work in this shared checkout.

## Approval gate

Ask exactly:

> Spencer, do you approve a maintenance window now to run exactly three consecutive P03 cycles using `./scripts/restore-two-spark.sh`, with the live two-Spark GLM API intentionally unavailable during each stop/start, immediate abort on the first nonzero exit or missing exact `TWO_SPARK_BASELINE_OK`, and no MCDMA startup or routing changes? Estimated interruption is 5-10 minutes per cycle (15-30 minutes total; inference, not yet measured). Yes or no?
