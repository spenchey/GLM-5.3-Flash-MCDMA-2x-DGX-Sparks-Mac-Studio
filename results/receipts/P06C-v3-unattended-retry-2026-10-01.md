# P06C protocol-v3 unattended retry receipt

Date: 2026-10-01

Linear issue: `MOT-3362`

Result: **BLOCKED — THROUGHPUT PASSED; FIRST SOFTWARE-FAULT SETUP MISSED ITS CONTROLLER DEADLINE**

The protocol-v3 throughput gate passed exactly as written. The campaign then
stopped at the first software-only stage because the adapter's cold setup was
still running when the controller's five-second `snapshot` deadline expired.
The actual five-second no-service wait was never reached. Orderly daemon loss,
fresh restart, and same-daemon reconnect were not run. No cable, interface,
route, firewall, driver, GLM, or API-routing change was made, and no `SIGKILL`
was used.

## Fixed preflight and provenance

- Project HEAD was the required clean commit
  `5ba344611c2d08ccbaf6b43a50a29c8b65cb7a1a`.
- Exact MCDMA base:
  `719219272c9ce6fc091b4eab6214eac510b5e387`.
- Safe-cleanup patch SHA-256:
  `57c5a86174920e392941fe0b1f3209377bb42954896f16c0d721769c57494839`.
- Protocol-v3 verification patch SHA-256:
  `630aeabb8bd9278e0efcf1c090355ce27d1bf09fcfffa0208c50e9b4b013059e`.
- Accepted manifest SHA-256:
  `b98ad1c623155ce0754f4b2f51fee82560766a0c8cff0d20e152c2ffeac4e23e`.
- Patched source hashes matched on Mac and Spark: `mcdma_bw.c`
  `df903c648fc0db486b88c061a37eceea31c9e85788ed6e73e4ad7597c20e8943`,
  `run_bw.py`
  `0f27d6cc86b0c1d628a97db3037d009c5944f0422f20ae17de420e03c06dcb5e`,
  and `verify_windows.h`
  `26731a52940927b08f905eb17462ee6c05cb0524f1b988ec31f46d8fe3f587a4`.

Both authoritative MCDMA checkouts were at the exact base and tracked-clean.
The isolated builds were:

| Host | New protocol-v3 `mcdma-bw` SHA-256 | Preserved prior SHA-256 |
| --- | --- | --- |
| Mac Studio | `6020d0b9a50088f8f492bb283d917c99d9910409d241160550294852d8d62946` | `de919c4906b62cb29a847c1a26b8fe965f054569edfe6f20ff0b9b97a5057d0f` |
| `<head-spark>` | `e778f36803feb3f04be214916f820ab2f8aca4b395e87c8aafd92b8de7a788c2` | `64d3b2f93022c506faefb12b2134e3bb79894d34fe4589f1d73ed41e34aeb8b9` |

Each prior executable was copied with its mode and owner to an adjacent,
hash-named backup. Each replacement was installed to a same-filesystem
candidate, hash-checked, then moved atomically into place. The installed RPC
daemon hashes were unchanged: Mac
`950c7cace2b30c8d45070d61938f4f7b6d80496ab961b4dbf4ed208e4645658b`
and Spark
`dd56f9f58242dc825cf8e9fefcbb4e99640afb76cc729d2a55ece0f4808acc1a`.

Hardware identity also matched the fixed plan: Mac build `26A428`,
`rdma_mcrdma1` GID index 0 at
`<redacted-gid>` with RoCE v2; `<head-spark>` `rocep1s0f0`
GID index 1 on `enp1s0f0np0` at
`<redacted-gid>`, RoCE v2, active MTU 4096. The separate
inter-Spark path remained `rocep1s0f1` / `enp1s0f1np1`, with distinct link
identities and `<private-ip>` / `<private-ip>`. Preflight found no test process,
socket, mailbox, tunnel, or port-18620 listener and sufficient disk on all
hosts. Both GLM containers were running, OOM false, restart count 0; local and
external APIs returned HTTP 200.

## Stage 1 — PASS

The accepted runner expanded to exactly 12 pairs: READ and WRITE, Mac and
Spark initiating, three measured repeats each. It ran from
`2026-10-01T22:49:49Z` to `2026-10-01T22:51:15Z` and exited zero with 12 JSON
logs, four CSV files, and no manifest errors.

Every one of the 24 measured endpoint rows and 24 warm-up endpoint rows had:

- protocol v3 and matching explicit 1,048,576-byte verification negotiation;
- `verified_bytes=1048576`, `mismatches=0`, `errors=0`, and `guard_ok=1`;
- exact 4 MiB request size, 8 GiB total, depth 1, one QP, one warm-up and one
  measured trial;
- one local verification proof row and one peer-propagated proof row per pair;
- both endpoint exits zero and `BW_DONE trials=2 failed=0 cleanup=0`.

Measured initiator payload rates were:

| Direction | Three Gbit/s measurements | Median | Range |
| --- | --- | ---: | ---: |
| Mac READ | 38.7284, 38.7634, 38.7356 | 38.7356 | 38.7284–38.7634 |
| Mac WRITE | 27.2390, 27.0057, 25.2012 | 27.0057 | 25.2012–27.2390 |
| Spark READ | 25.0042, 23.5076, 26.0749 | 25.0042 | 23.5076–26.0749 |
| Spark WRITE | 38.8765, 38.8712, 38.8773 | 38.8765 | 38.8712–38.8773 |

These are bounded RDMA payload rates, not model-inference rates. Immediately
after Stage 1, no benchmark or daemon remained, port 18620 had no listener,
both GLM containers were still running with OOM false/restarts 0, and both API
checks were HTTP 200.

## First software stage — BLOCKED before the fault

The exact controller invocation used the accepted helper and live adapter:

```text
P06_LIVE_STATE=<raw>/fault-no-service P06_LIVE_SCENARIO=no-service \
  python3 scripts/tests/p06-fault-helper.py no-service-timeout \
  --adapter <repo>/scripts/tests/p06-live-adapter.py
```

It exited exactly `3` with:

```text
DEADLINE_FAILURE: adapter operation snapshot exceeded 5.0s
```

This was the controller's five-second timeout for the initial `snapshot`
adapter operation, not the registered five-second no-service reply wait.
`snapshot` calls `ensure_started()`, which cold-compiles two helpers and then
starts listener, tunnel, connector, and waits for link readiness. At expiry,
only Spark listener PID `346218` had started. No connector or tunnel was left.
Because the adapter saves `spark_pid` only after all cold setup completes, the
cleanup process reloaded state without that PID and reported the Spark process,
socket, and listener as residue rather than shutting it down. The no-service
request was never staged, so no counter or timeout result may be claimed.

The smallest safe repair is to make cold setup an explicit separately bounded
operation before the five-second snapshot, and to persist each owned PID
immediately after it is started. Cleanup can then always identify and shut down
a partially started session. The fixed five-second duration must remain on the
actual no-service reply wait; it must not be weakened or conflated with setup.
This receipt does not implement or retry that repair.

## Orderly cleanup and final health

The interrupted listener was stopped only through its normal UNIX control
socket. `SHUTDOWN` returned exact `BYE`; PID `346218` exited; its RPC socket,
shared mailbox, and `127.0.0.1:18620` listener disappeared. Its log ended with
`every verbs object destroyed, exiting`. No signal escalation or forced kill
was used.

Final proof found:

- zero `mcdma-rpcd`, `mcdma-bw`, P06 helper, or controller processes;
- no Mac/Spark RPC socket, control socket, mailbox, or port-18620 listener;
- all exact Mac and Spark test/build temporary trees absent;
- both installed protocol-v3 bandwidth hashes and both preserved prior hashes
  still exact; RPC daemon hashes unchanged;
- both MCDMA source checkouts still at the exact clean base;
- Mac RDMA enabled and the planned CX5 port/GID still active; Spark port still
  active with MTU 4096;
- head and worker GLM running, OOM false, restart count 0; local and external
  API HTTP 200.

Idle-loss/fresh-restart, reconnect, cable testing, and P06D were not run.

## Private evidence

Raw evidence is intentionally ignored because native benchmark logs contain
QP identifiers, keys, and addresses:

```text
results/raw/P06C-v3-retry-2026-10-01/
```

The 44-file SHA-256 inventory is `raw-sha256.txt`, whose SHA-256 is
`0a84444a84e0a8ee51cb2b28fe1df1bf7a7b3ee4437b5c4183f9002dbf2a2b5d`.
Key evidence hashes are:

- Stage 1 gate matrix: `90d9b2bdde49396ea1813fe0b9a7cca0a8caa42b1d413284622c0ab812962bb9`
- Stage 1 manifest: `cea6e20adced541994830183568ea7582c9daf64f5c2aa7d50749c171e4b6dce`
- no-service error: `7efd07965787afb9241e6d1576f182c765774d0f287a6555959e943ac225ab54`
- interrupted-state cleanup report: `88f05dc2e30a20d335c3aa7da04c6f09f4a064ba2fe89681198fcb0a62131a52`
- listener teardown log: `fdb6b66f1512b163082689aa396a3cae0fa00987399146f53c79e8b381b1f1f2`
- final cleanup/health: `43c3175d2fed42b730eaa0c59046d50c1ae7e84171c5e964a8884a7acc7885ae`

P06C remains blocked. This is not a P06 or P06D pass.
