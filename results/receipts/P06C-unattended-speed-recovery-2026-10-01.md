# P06C unattended speed and recovery receipt

Date: 2026-10-01

Linear issue: `MOT-3362`

Result: **BLOCKED — STAGE 1 VIOLATED THE WRITTEN ALL-ROWS VERIFICATION GATE; STAGE 2 ADAPTER FAILED BEFORE DAEMON START**

P06C stopped at the first two fail-closed blockers. Twelve throughput pairs
completed without transport, completion, guard, or data mismatch errors, but
six measured READ sender rows reported `verified_bytes=0`. The written gate
requires every measured endpoint row to report exactly 1,048,576, so Stage 1
cannot be called a pass. The five-second no-service stage was then started in
error and failed before any daemon started because the reviewed live adapter's
SSH shell invocation leaked `0022` into compiler stdout. No later fault stage
ran. No cable, route, interface, firewall, driver, GLM, or API-routing change
was made, and no SIGKILL was used.

## Accepted source and preflight

The run used repository commit
`69db1c802a7baed376a65098f0153bee827db9d5` and a new ignored tree
materialized from exact MCDMA base
`719219272c9ce6fc091b4eab6214eac510b5e387` with:

- P06C0 patch SHA-256
  `57c5a86174920e392941fe0b1f3209377bb42954896f16c0d721769c57494839`;
- P06C1 patch SHA-256
  `69438f5992f256f512eeccc4bf70bad5fd4a2a6f97a33e367c286628d5de91e5`;
- manifest SHA-256
  `2ee798327842100e6f6de2ea5dd813cfaedeafbffa59a09e28c8a936cd77a57a`;
- patched runner SHA-256
  `cecf329dec1adcadac50432ed4ea58f5857266f14c29198dfea5a63f62c4ff23`;
- patched C/header SHA-256 values
  `3e11e61a3cbdbcbcb74b1d4b9cb77550e76ca1adde9cf410cd04deddf6417aa9`
  and `cc18a4f0a45c11815cf77d1023fb8b5dcc670d10e70811e7ce45298acba0e53e`.

Fresh isolated builds matched the accepted binary hashes: Mac Studio
`8b07c05698a5ea9fd7b37d7dc2df0c55a42b7d121018262185ef7c5174062081`
and Spark
`7f3ae470ee5d011beacdb7bcd390529b087784731147b3287d7c0c6d4bebe524`.
Installed binaries were not replaced.

The immediate identity gate proved:

- Mac `spencers-Mac-Studio.local`, build `26A428`, RDMA enabled,
  `rdma_mcrdma1` active at MTU 4096 with RoCE-v2 GID index 0, and `mcrdma1`
  up;
- head `<head-spark>`, `rocep1s0f0` active at MTU 4096, RoCE-v2 GID index 1,
  and `enp1s0f0np0` up;
- worker `<worker-spark>` retained the distinct inter-Spark link
  `enp1s0f1np1` up;
- no MCDMA process, RPC socket, tunnel, or port-18620 listener existed;
- both `glm53-flash-tf` containers were running, OOM false, restart count 0,
  and the head model API returned HTTP 200.

The exact dry run expanded to 12 pairs.

## Stage 1 — measured result and blocking gate

All 12 pairs completed. Both endpoints returned zero for every pair; runner
errors were empty; protocol v2 and explicit requested/effective verification
of 1,048,576 bytes were agreed; every measured row reported `errors=0`,
`mismatches=0`, and `guard_ok=1`; all CSV hashes match the manifest.

Initiator payload-rate medians and observed ranges were:

| Direction | Median Gbit/s | Range Gbit/s |
| --- | ---: | ---: |
| Mac READ | 38.6750 | 38.6484–38.6861 |
| Mac WRITE | 24.7766 | 23.4128–27.2050 |
| Spark READ | 25.2646 | 25.0123–25.3211 |
| Spark WRITE | 38.7653 | 38.6300–38.7693 |

There were 24 non-warmup endpoint rows. Eighteen reported
`verified_bytes=1048576`; the six READ responder/sender rows, one for each
Mac/Spark READ direction and repeat, reported `verified_bytes=0`. The exact
12-pair/24-row matrix is retained in `stage1-row-matrix.json` and marks every
actual receiver.

This zero is explained by the patched binary, not by an inferred hardware
failure. In `run_initiator_trial()`, the READ initiator receives the data and
runs `verify_windows()`. In `run_responder_trial()`, the READ responder sends
data and deliberately executes verification only when `op != READ`; its local
verified count therefore remains zero. WRITE verification runs on the
responder and is copied back into the initiator's result through the `DONE`
message, which is why both WRITE rows report one MiB.

The current READ `COMPLETE trial=... ok=...` message does not carry the
receiver's verified-byte count. Therefore the current protocol cannot
truthfully populate that count in the READ sender's row. One of two separately
reviewed changes is required before another run:

1. extend and version the protocol so the READ receiver sends its verified
   bytes/mismatch result to the sender and the runner validates the propagated
   value; or
2. revise the acceptance contract explicitly to require exactly one MiB on
   each actual receiver row while requiring zero on a READ sender row.

This receipt does neither and does not reinterpret the written gate.

## Stage 2 — exact adapter failure

The attempted command was:

```text
P06_LIVE_STATE=<raw>/fault-no-service P06_LIVE_SCENARIO=no-service \
  scripts/tests/p06-fault-helper.py no-service-timeout \
  --adapter <repo>/scripts/tests/p06-live-adapter.py
```

It returned exit code 1 with:

```text
CONTRACT_FAILURE: adapter operation snapshot exited 1: P06_LIVE_ADAPTER_FAILURE: mailbox helper compile emitted unexpected output
```

Exact-argv reproduction proved the cause. `Remote.shell()` passes SSH the
separate arguments `sh`, `-c`, and `umask 077; cc ...`. OpenSSH concatenates
them into an unquoted remote command, so the remote shell executes
`sh -c umask 077` followed by `cc ...`. Bare `umask` prints exactly `0022\n`.
Compilation returns 0 with empty stderr, but `Remote.compile()` correctly
rejects any compiler stdout/stderr.

The same argument-quoting defect caused the adapter's cleanup check
`sh -c test ! -e /tmp/p06cns-501/control.sock` to return 1 even though the
socket was absent. State recorded `tunnel control socket survived stop`, but
direct inspection proved the exact socket path absent, no process owned it,
and no port-18620 listener existed. No RPC daemon had started. Normal cleanup
ran during helper exit and once more as a cleanup-only recheck. No SIGKILL flag
was ever set; the recheck listed no surviving child. Confirmed inactive
temporary trees were then removed.

Orderly daemon-loss/fresh-restart and same-daemon reconnect were **not
started**. No cable stage was run.

## Final cleanup and health

- Mac installed bandwidth/RPC hashes remained
  `de919c4906b62cb29a847c1a26b8fe965f054569edfe6f20ff0b9b97a5057d0f`
  and `950c7cace2b30c8d45070d61938f4f7b6d80496ab961b4dbf4ed208e4645658b`.
- Spark installed bandwidth/RPC hashes remained
  `64d3b2f93022c506faefb12b2134e3bb79894d34fe4589f1d73ed41e34aeb8b9`
  and `dd56f9f58242dc825cf8e9fefcbb4e99640afb76cc729d2a55ece0f4808acc1a`.
- Mac, head, worker, and local controller all had zero owned processes; all
  three hosts had zero port-18620 listeners; RPC and tunnel sockets were
  absent; isolated binary/helper temporary trees were absent.
- Head and worker GLM stayed running, OOM false, restart count 0; model API
  stayed HTTP 200.

Final cleanup/health log SHA-256:
`81f36e134aadfd1e0b529f53af322c5d79c18632cb36d8edf5e974cf210afcb2`.

## Raw evidence

Owner-only ignored evidence is under
`results/raw/P06C-live-final-2026-10-01/`. Key SHA-256 values:

- raw inventory: 216 entries; inventory SHA-256
  `2a38519b0b26025e88e58853a370bb348cc8654306ef43c0d978dfdaa8b5f411`;
- throughput manifest:
  `eea19e120a3b28edd22cf1e3f90cc13866e4737994bbe4875b4df8d8c93d1271`;
- exact row matrix:
  `7bb94d6eee31b147cf92485f1c593a4f1639154310831da0684cf0ed6eb5dcfe`;
- no-service failure log:
  `c0d7c963baa4b3900ca78cb1fea5290f91d0aae61c201d6bf29b4af9402afc12`;
- exact compile-argv audit:
  `7cccd3243070765af84fb44e53c38d598c0e5ec5cd67f8f2bc71adf9da058801`;
- exact tunnel-check audit:
  `af8aff0b05e56b12d1139294e3f4bb4dc915db3e06181773cc202c09b0eca167`;
- final cleanup/health:
  `81f36e134aadfd1e0b529f53af322c5d79c18632cb36d8edf5e974cf210afcb2`.

P06C remains blocked. Do not mark it Done, do not run P06D, and do not retry
until the written verification gate and live-adapter quoting defect receive
separate review and repair.
