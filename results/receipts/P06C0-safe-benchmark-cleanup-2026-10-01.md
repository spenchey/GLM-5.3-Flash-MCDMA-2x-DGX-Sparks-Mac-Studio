# P06C0 safe benchmark cleanup receipt

Date: 2026-10-01

Linear issue: `MOT-3364`

Result: **PASS — REPRODUCIBLE OFFLINE PATCH; NO LIVE BENCHMARK STARTED**

This change removes the forced-kill path that blocked P06C. It is a plain,
reviewable patch against the exact pinned MCDMA revision. The authoritative
Mac Studio checkout and installed programs were not changed.

## Pinned input and reproducible output

- MCDMA base: `719219272c9ce6fc091b4eab6214eac510b5e387`
- patch: `patches/mcdma/0001-safe-benchmark-cleanup.patch`
- patch SHA-256:
  `57c5a86174920e392941fe0b1f3209377bb42954896f16c0d721769c57494839`
- hash manifest: `patches/mcdma/safe-benchmark-cleanup-manifest.json`
- preparation helper: `scripts/prepare-mcdma-safe-benchmark.sh`
- offline harness: `tests/test-mcdma-safe-benchmark-cleanup.sh`

The helper refuses a wrong revision, tracked source changes, a pre-existing
destination, a patch-hash mismatch, any base-file hash mismatch, or any
patched-file hash mismatch. It archives the exact base into a new isolated
directory, runs `git apply --check`, applies the patch there, verifies every
patched hash, and writes a copy of the manifest into the prepared directory.

Production file hashes:

| File | Base SHA-256 | Patched SHA-256 |
|---|---|---|
| `tools/native_cross_host.py` | `e38668dacdbedded6c4047af93f8615855bacae6b515d010f6cd0b38d330ba48` | `2ce10a1506dd72f44f6aa1f7fe6917c3643f8034469b16d78e1f92d0cbbeba1f` |
| `benchmarks/run_bw.py` | `5676178b1c6f679d2cf2bcf054f96da34edbf9b04e485f30ac124f0fffe5fbff` | `f5af93517e57eaf327ceefae3f9898783d695a6e3ba2554a91f906c9a044fbcf` |

The manifest also pins the base and patched hashes of both modified upstream
test files.

## Cleanup behavior proved

`Endpoint.stop()` now performs only these bounded steps:

1. close endpoint stdin;
2. wait up to 15 seconds;
3. if still alive, send `SIGTERM` with `terminate()`;
4. wait up to five more seconds;
5. if still alive, raise `EndpointCleanupError` naming the surviving PID.

There is no `process.kill()` fallback. A surviving endpoint is reported and
the benchmark fails; it is never silently accepted and the next trial cannot
start.

`run_bw.py` installs handlers for `SIGTERM` and `SIGHUP` before the first
endpoint is created. Each handler raises the same interruption used by the
existing per-pair `finally` block, so every already-created endpoint receives
the bounded cleanup above. Previous signal handlers are restored before
`main()` returns. Cleanup errors are recorded in the run manifest and force
the exact nonzero result `1`.

## Deterministic offline tests

The patch adds upstream tests for:

- normal endpoint exit;
- first 15-second timeout followed by successful `SIGTERM` cleanup;
- a child that survives both bounded waits;
- exact wait calls of 15 seconds and five seconds;
- `SIGTERM` and `SIGHUP` reaching cleanup after both endpoints exist;
- cleanup failure producing exact `run_bw` exit `1` and retaining the PID;
- `process.kill()` never being called in any cleanup branch.

Validation completed in ignored isolated copies only:

```text
./tests/test-mcdma-safe-benchmark-cleanup.sh <exact-base-checkout>
  23 tests passed
  PASS: exact-base patch, hashes, signals, nonzero cleanup failure, and no forced kill

python3 -m unittest -v \
  tests.test_native_endpoint tests.test_bw_tools tests.test_bw_guard \
  tests.test_bw_payload tests.test_bw_payload_trial tests.test_ndp_neighbor \
  tests.test_native_observer_marker
  51 tests passed

bash -n scripts/prepare-mcdma-safe-benchmark.sh \
  tests/test-mcdma-safe-benchmark-cleanup.sh
  passed
```

`PYTHONDONTWRITEBYTECODE=1` was set for Python validation. A final process
check returned `no_test_child_survives`; no tracked or ignored `.pyc` cache was
left in the project test directories.

## Authoritative checkout and live-state proof

The final read-only authoritative check returned:

```text
commit=719219272c9ce6fc091b4eab6214eac510b5e387
tracked_status_lines=0
tools/native_cross_host.py=e38668dacdbedded6c4047af93f8615855bacae6b515d010f6cd0b38d330ba48
benchmarks/run_bw.py=5676178b1c6f679d2cf2bcf054f96da34edbf9b04e485f30ac124f0fffe5fbff
```

Final GLM health was unchanged:

```text
head:   running=true oom=false restarts=0
worker: running=true oom=false restarts=0
API:    HTTP 200
```

No benchmark, MCDMA daemon, tunnel, port 18620 listener, RDMA transfer,
interface/route/firewall/driver action, cable action, GLM restart, or API
routing change was performed. The isolated raw copies and broader test tree
remain under ignored `results/raw/P06C0-safe-benchmark-cleanup-2026-10-01/`
for private verification.

P06C may resume only after this patch is independently reviewed and its exact
patch hash is registered for the live preparation step.
