# P02 rollback controller dry-run receipt

Date: 2026-10-01
Linear issue: `MOT-3328`
Result: **PASS**

P02 added one rollback command that stops the current recipe-managed experiment,
starts the exact pinned two-Spark service, and withholds
`TWO_SPARK_BASELINE_OK` until every restore gate passes. This receipt covers
code, isolated shell tests, and dry-run only. No SSH command was executed; no
container or route was stopped, restarted, replaced, or reconfigured.

## Command

```bash
./scripts/restore-two-spark.sh
```

The real command fails closed in this order:

1. verify the pinned recipe commit, clean tracked recipe files, worker/MTP
   selection, image ID, and model revision on both Sparks;
2. stop the recipe-managed experiment and start the pinned service with
   `DRAFTER=mtp PREPARE=0`;
3. require both containers to be running, healthy when a Docker healthcheck is
   present, not OOM-killed, at restart count zero, and on the pinned image ID;
4. require the API to expose exactly `GLM-5.3-Flash-EXL3`; and
5. require one normally stopped answer whose visible content is exactly
   `GLM_READY` with the already-proven 256-token budget.

Only after all five gates pass does the controller print
`TWO_SPARK_BASELINE_OK`.

## Verification

Commands run:

```bash
bash -n scripts/*.sh tests/*.sh
./tests/test-config.sh
./tests/test-restore-two-spark.sh
./scripts/restore-two-spark.sh --dry-run
```

Observed results:

```text
PASS: pinned inputs and private-data guard
PASS: rollback dry-run is inert and success marker is health-gated
DRY-RUN: no SSH command will be executed and no host will be changed.
PLAN 1 through PLAN 7 listed the exact verify, stop, start, two-container,
API-model, and fixed-answer commands.
DRY-RUN COMPLETE: the success marker is intentionally withheld.
```

The focused test replaces `ssh` with a local sentinel. It proves the dry-run
never invokes that sentinel, proves the success path prints the marker once,
then separately injects OOM, restart, unhealthy, wrong-image, wrong-model, and
wrong-answer failures. Every injected failure exits nonzero without printing
the marker.

Evidence hashes:

- dry-run output SHA-256:
  `91edcf5f22fde28059f7dd95240f93ce8cb81159e6151c4d142982b0ccb11035`
- focused-test output SHA-256:
  `4a179385e77edfcc2bf2c38b49d57c7f1b18eac63250437e70b1b8a61a9e1a14`
- controller SHA-256:
  `3996b6a541d782265f0792cf21817a8f7935787a1c035690e8167b8896b58dba`
- focused-test SHA-256:
  `8a2dd554aaa4eef66179fca9763540dc9c9ec4da3bc37a7f11b525cec149fb45`
- implementation base commit:
  `d0a9bd3cbd3d7df1d339c0bbeaaec50caca54448`
- implementation commit:
  `2ec88f2fd2f8ad904dd4267208d8c6510590203a`

P02's stated pass proof is satisfied. Live rollback rehearsal remains P03 and
still requires Spencer's approved maintenance window.
