# P06B bounded fault-helper receipt

Date: 2026-10-01

Linear issue: `MOT-3361`

Result: **PASS — OFFLINE HELPER ONLY; NO HARDWARE TEST STARTED**

This receipt covers the separate P06 controller and deterministic local mocks.
It does not claim a live timeout, daemon-loss, reconnect, cable, throughput, or
GLM result. The helper did not start an MCDMA daemon, open port 18620, contact
either remote host, use RDMA, move a cable, change an interface/driver, or
stop/restart GLM.

## Owned files

- `scripts/tests/p06-fault-helper.py`
- `tests/test-p06-fault-helper.py`
- `results/receipts/P06B-bounded-fault-helper-2026-10-01.md`

Final SHA-256 values for the executable and its test suite:

- helper: `a3d9ba1345ef694551dacb5f2bc5b05bc788c78b55047af3060ccb458232e24b`
- tests: `de8115379d6cb9d4812b006a05b9bc64a7edc83f796a5b402924fb93370f9daa`

The P05 mailbox helper and the P06 preflight receipt were not changed.

## Enforced modes

The controller has four explicit modes:

1. `no-service-timeout` stages the pinned 32 KiB request hash, waits five
   seconds, accepts only an explicit timeout with no reply, then requires an
   unchanged generation, exactly one new Mac call, no Spark call, and no
   failure-counter change.
2. `idle-daemon-loss` begins only while idle and up, accepts only orderly Mac
   connector teardown with verbs cleanup and no `SIGKILL`, then requires link
   down, the same live Spark listener, unchanged counters/generation, and a
   disconnected service.
3. `same-daemon-reconnect` drops and restores only the control path, requires
   both daemon PIDs to remain unchanged, bounds reconnect to 30 seconds,
   requires an exact one-step generation advance, and byte-hash checks the
   first recovered call before accepting exact counter changes.
4. `cable-window` stages a call, prints the registered safe-disconnect prompt,
   requires the exact `DISCONNECT` response within 30 seconds, accepts only an
   explicit timeout/failure or a fully hashed reply, requires the exact
   `RECONNECTED` response, bounds same-daemon recovery to 30 seconds, and
   hash-checks a fresh recovery call.

The fixed 32 KiB hashes come from the proven P05 pattern:

- request: `66c60468130441e8b45cdf0e7e79ae9e74c08bf3c84d05e4d40ded131e363b9c`
- reply: `cf7ef6b9110e2389d0140a03d32bf910de7a02b6ea8f39283cc237eed1776096`

The later P06C/P06D runner must supply a separately reviewed executable
adapter. Every adapter operation emits exactly one JSON object. The controller
validates link state, generation, daemon PIDs/aliveness, call/failure counters,
in-flight/reply state, service state, hashes, explicit outcomes, and cleanup.
The adapter operations and required cleanup result are printed by `--help`.

## Fail-closed and cleanup behavior

- Exit `0`: the selected mode and cleanup both passed.
- Exit `1`: first state, counter, generation, hash, or adapter-contract
  mismatch.
- Exit `2`: invalid invocation or unsafe adapter selection.
- Exit `3`: a fixed deadline expired.
- Exit `4`: the operator prompt was missing, late, or not exact.
- Exit `5`: cleanup proof failed.
- Exit `130`: interruption after cleanup was attempted.

Cleanup handlers are registered before the first adapter child starts. On
success, failure, signal, or timeout, the controller sends only `SIGTERM` to
an active adapter process group and allows three seconds for orderly exit. It
never escalates to `SIGKILL`. It then runs the adapter's bounded cleanup and
requires exactly `clean=true`, `sigkill_used=false`, and
`children_alive=[]`. A child that ignores `SIGTERM` is an explicit cleanup
failure, not permission to force-kill or continue.

## Offline test proof

`tests/test-p06-fault-helper.py` uses only a temporary local state file and
local mock adapter. The mock creates one local `sleep` child so every case can
prove that cleanup removes it. The suite proves:

- the real five-second no-service deadline and exit `0`;
- accepted request/reply hashes and rejected request/reply hashes;
- first counter mismatch failure and exact exit `1`;
- adapter-deadline termination and exact exit `3`;
- orderly idle daemon loss;
- same-PID reconnect and exact generation `1 -> 2`;
- generation skip rejection;
- both exact operator prompts, wrong-token abort, and exit `4`;
- cleanup-failure reporting and exit `5`;
- invalid invocation and exit `2`;
- no mock child survives successful or failed cleanup.

Validation commands:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  scripts/tests/p06-fault-helper.py tests/test-p06-fault-helper.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/test-p06-fault-helper.py
bash -n scripts/mcdma-control-tunnel.sh tests/test-mcdma-control-tunnel.sh
cc -std=c11 -O2 -Wall -Wextra -Werror -fsyntax-only \
  scripts/tests/mcdma-rpc-mailbox.c
./tests/test-config.sh
./tests/test-restore-two-spark.sh
git diff --check
```

The final suite result was 12 tests passed. Existing local configuration and
rollback tests also passed. The existing live-SSH tunnel integration test was
not run because P06B is expressly offline-only.

## Remaining gate

P06B supplies enforcement logic, not the live adapter or hardware result.
P06C must implement/review the adapter against the pinned daemons and run only
the unattended stages. P06D still requires Spencer's exact cable approval and
presence. Neither later stage may weaken these deadlines, hashes, counter
checks, same-PID/generation requirements, prompt tokens, or cleanup contract.
