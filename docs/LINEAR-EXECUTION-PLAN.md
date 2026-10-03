# Linear execution plan

> Historical decomposition used to reach the current implementation. The
> authoritative remaining work and completion state are in `PROJECT-PLAN.md`;
> the operational result is in `THREE-MACHINE-RUNBOOK.md`.

## Project

**Name:** GLM 5.3 Flash — Mac Studio + Dual Spark MCDMA

**Outcome:** One GLM-5.3 Flash request must use the Mac Studio for a real
TensorFold Metal stage, both DGX Sparks for the remaining TensorFold CUDA
stage, and MCDMA for the Mac/Spark boundary. The old two-Spark service remains
preserved as manual rollback material, but D-025 supersedes the early rule that
restored it after every test: the three-machine service is the deployment goal.

The parent Linear issue owns the completion decision. A child issue is complete
only when its stated receipt exists and its exact check passes. Prepared code,
a live link, or a running process is not proof of the whole project.

## Working rules

- Work in the dependency order below. Do not start a blocked issue.
- One issue changes one thing and produces one receipt.
- Do not run two issues at once when they edit the same file.
- Read-only checks and isolated unit tests may run unattended.
- Stopping the live service, changing a driver, pulling a cable, rebooting, or
  switching API traffic requires Spencer's approval for that maintenance
  window.
- Every failed attempt is appended to `docs/EXPERIMENT-LOG.md`; never erase it.
- Each implementation issue records its exact commit, commands, output hashes,
  result, and rollback in `results/receipts/`.
- DFlash2 stays excluded.

## Atomic issues

### P00 — Capture the exact live TensorFold build

- **Depends on:** none
- **Owner:** operations agent
- **Mode:** unattended, read-only
- **Scope:** `UPSTREAM.lock`, one new receipt, and an ignored source snapshot
- **Do:** extract the TensorFold source, package version, image ID, patch label,
  start command, model revision, and hashes from the running two-Spark image.
- **Do not:** stop or replace either live container.
- **Pass proof:** the receipt can recreate the source identity and
  `./scripts/status.sh && ./tests/test-config.sh` still pass.

### P01 — Measure the unchanged two-Spark baseline

- **Depends on:** P00
- **Owner:** benchmark agent
- **Mode:** unattended, read-only
- **Scope:** benchmark helper plus one receipt; raw output stays ignored
- **Do:** run the fixed prompt set with five fixed seeds, retain raw output
  hashes, and label page-migration slow runs instead of hiding them in averages.
- **Pass proof:** the receipt contains output hashes, TTFT, token latency, rate,
  end-to-end time, memory high-water marks, and service health for every run.

### P02 — Build the rollback command and dry-run it

- **Depends on:** P00
- **Owner:** operations coding agent
- **Mode:** unattended code work only
- **Scope:** controller scripts and shell tests only
- **Do:** add one command that can stop an experiment, restore the pinned live
  service, and print `TWO_SPARK_BASELINE_OK` only after health and fixed-answer
  checks pass. Add a dry-run that changes nothing.
- **Pass proof:** `bash -n` and shell tests pass; dry-run lists exact actions and
  makes no host change.

### P03 — Rehearse rollback three times

- **Depends on:** P01, P02
- **Owner:** operations agent
- **Mode:** owner-approved maintenance window
- **Scope:** receipt only
- **Do:** perform three bounded stop/check/restore cycles with the new command.
  Abort the project on the first failed restore.
- **Pass proof:** three consecutive receipts end in
  `TWO_SPARK_BASELINE_OK`, with both containers healthy and their fixed answer
  unchanged.

### P04 — Prove all three machines are ready

- **Depends on:** P00
- **Owner:** operations agent
- **Mode:** unattended, read-only
- **Scope:** one receipt
- **Do:** record exact Mac/Spark OS and software revisions, TensorFold and MLX
  versions, MCDMA driver and daemon builds, loaded driver state, free disk, free
  memory, link identity, and the intended process/container owners.
- **Pass proof:** every required version and resource has an observed value;
  missing or mismatched items are explicit failures, not assumptions.

### P04A — Build and offline-test the Linux MCDMA service

- **Triggered by:** P04's observed missing Linux `mcdma-rpcd` binary
- **Owner:** operations coding agent
- **Mode:** unattended, isolated build only
- **Scope:** a user-owned install directory plus one receipt; no system install
- **Do:** build the exact pinned MCDMA source on the head Spark, run its offline
  test suite, install the resulting binary into a user-owned project/state
  directory, and record its version and SHA-256.
- **Do not:** use `sudo`, write to `/usr/local`, start the daemon, change a
  network interface, or stop/restart either live GLM container.
- **Pass proof:** the build and offline tests pass, the installed binary's
  version/hash are recorded, the service is not running, and the live GLM API
  plus both containers remain healthy with zero OOM or restart change.

### P04B — Secure the MCDMA control endpoint through SSH

- **Depends on:** P04A
- **Owner:** network operations agent with Spencer
- **Mode:** unattended, isolated control-tunnel test only
- **Scope:** loopback SSH-tunnel helper, tests, and one receipt; host network
  and RDMA interfaces remain unchanged
- **Do:** keep MCDMA data on the direct `mcrdma1` / `rocep1s0f0` RDMA link;
  bind the Spark daemon control listener to `127.0.0.1:18620`; use one
  authenticated SSH local-forward bound to Mac `127.0.0.1:18620`; document
  process ownership, startup order, failure behavior, and exact orderly
  shutdown; prove the tunnel with an isolated loopback fixture before using the
  hardware daemon.
- **Do not:** add IPv4 to the RDMA interfaces, bind the control service or SSH
  forward to a non-loopback address, change a firewall/Tailscale ACL, change
  the inter-Spark documentation subnet `192.0.2.0/24`, or touch the live GLM
  containers.
- **Pass proof:** the isolated fixture is reachable through Mac loopback only;
  neither endpoint exposes `18620` on LAN or tailnet addresses; the tunnel
  fails closed when stopped; cleanup leaves no listener, control socket, or SSH
  process; and a repeated P04 readiness check has no missing daemon
  build/address/port values. P04 then becomes complete.

### P05 — Prove `mcdma-rpcd` on the real Mac/Spark link

- **Depends on:** P04
- **Owner:** transport agent
- **Mode:** unattended only if no driver or cable change is needed
- **Scope:** MCDMA config/tests plus one receipt
- **Do:** use the real daemon path for checksum-verified 32 KiB and 2 MiB
  request/reply transfers, record counters, bind control only to the private
  point-to-point link, and stop with orderly shutdown.
- **Pass proof:** both sizes match byte-for-byte, counters move by the expected
  amount, control is not exposed to other networks, and no daemon remains.

### P06 — Measure both link directions and fault recovery

- **Depends on:** P05
- **Owner:** completion gate; work is split into P06A-P06D plus the observed
  P06B1, P06C0, P06C1, and P06C2 repair gates
- **Mode:** mixed; P06D alone is owner-assisted
- **Scope:** no direct implementation work; close only after all eight children
- **Do:** use the preflight receipt as the fixed campaign contract. Do not let
  an agent skip a child or combine the physical cable test with preparation.
- **Pass proof:** P06A, P06B, P06B1, P06C0, P06C1, P06C2, P06C, and P06D are
  Done; the final receipt has exact one-MiB verification, measured rates,
  declared timeout behavior, clean recovery, no forced kill, no orphan process,
  and unchanged GLM service health.

### P06A — Build and pin the Spark bandwidth tool

- **Triggered by:** the P06 preflight found no pinned Spark `mcdma-bw`
- **Owner:** transport build agent
- **Mode:** unattended, offline build only
- **Scope:** the pinned user-owned MCDMA install tree and one receipt
- **Do:** confirm the source commit and clean tree; build `mcdma-bw`; install it
  only in the user-owned pinned MCDMA checkout under `install/bin/mcdma-bw`;
  record compiler, command, version, SHA-256, mode, tests, disk, and GLM health.
- **Do not:** use `sudo`, write to `/usr/local`, start a daemon, benchmark,
  open port 18620, change a network interface, touch a cable, or restart GLM.
- **Pass proof:** the exact executable is hashed and passes offline checks; no
  MCDMA process/listener remains; both containers and HTTP 200 remain healthy.

### P06B — Add the bounded MCDMA fault-test helper

- **Triggered by:** the P06 preflight found no bounded timeout/fault helper
- **Owner:** transport coding agent
- **Mode:** unattended, offline tests only
- **Scope:** a separate helper, deterministic tests, and one receipt
- **Do:** implement explicit modes for a five-second no-service timeout, clean
  idle daemon loss, same-daemon reconnect, and a 30-second operator-prompted
  cable window. Fail on the first unexpected counter/state and always clean up.
- **Do not:** start hardware daemons, transfer over the live link, touch a
  cable or interface, use `SIGKILL`, or restart GLM.
- **Pass proof:** compile/lint/tests prove deadlines, exits, hashes, generation
  checks, aborts, and cleanup; P05 remains unchanged and GLM stays healthy.

### P06B1 — Fix remote command quoting and cleanup proof

- **Triggered by:** the first P06C software-fault attempt exposed an SSH
  argument-boundary defect: the remote shell ran a bare `umask`, emitted
  `0022`, and also made an absent tunnel socket look present
- **Owner:** transport coding agent
- **Mode:** unattended, offline tests only
- **Scope:** the P06 live adapter, deterministic fake-SSH tests, and one receipt
- **Do:** pass every remote script as one safely quoted `sh -c` payload while
  preserving stdin-fed source, exact exit codes, and sanitized stdout/stderr;
  test semicolons, spaces, metacharacters, compilation, `test ! -e`, failure
  exits, missing-socket success, and active-socket safety.
- **Do not:** contact a live host, start a daemon, change an installed binary,
  touch networking/cables, stop GLM, or run a fault stage.
- **Pass proof:** the original `0022` failure reproduces before the fix; all
  new and prior adapter tests pass afterward; a missing socket is accepted and
  an active socket is never falsely removed. P06C remains blocked until this
  child is Done.

### P06C0 — Remove forced-kill cleanup from the MCDMA benchmark

- **Triggered by:** P06C pre-live review found a reachable `process.kill()`
  fallback and missing `SIGTERM`/`SIGHUP` cleanup routing
- **Owner:** transport coding agent
- **Mode:** unattended, isolated patch and offline tests only
- **Scope:** a reproducible patch against MCDMA commit `719219272c9c`, focused
  tests, preparation helper, and one receipt; authoritative checkouts unchanged
- **Do:** remove forced-kill escalation; after stdin close, bounded wait,
  `SIGTERM`, and a second bounded wait, fail and report the surviving PID;
  route `SIGTERM` and `SIGHUP` through the same bounded `finally` cleanup;
  test success, timeout, signals, failure reporting, and never-called `kill()`.
- **Do not:** run a benchmark/daemon/RDMA transfer, open port 18620, touch
  interfaces/drivers/cables, patch an authoritative checkout, or restart GLM.
- **Pass proof:** the patch applies cleanly to the exact commit; all old/new
  offline tests pass; no tested path calls `kill()`; signals reach cleanup;
  surviving children cause nonzero failure; no test process remains; GLM is
  unchanged. This prerequisite must stay Done before P06C1 or P06C runs.

### P06C1 — Make MCDMA verify exactly the requested bytes

- **Triggered by:** the first P06C Stage 1 run requested a 1,048,576-byte
  verification sample but every row reported 12,288 because `build_windows()`
  capped one resident slot at three 4,096-byte windows
- **Owner:** transport coding agent
- **Mode:** unattended, isolated patch/build/offline tests only
- **Scope:** a reproducible patch against exact MCDMA base `719219272c9c` plus
  P06C0, deterministic window tests, isolated binaries, and one receipt
- **Do:** make an explicit `--verify-bytes` value require at least 24 bytes,
  eight-byte alignment, and a value no greater than agreed resident capacity;
  verify it exactly or fail before trial traffic. When the flag is omitted,
  report auto mode and use exactly the smaller of one MiB and agreed resident
  capacity, still requiring at least 24 bytes. Negotiate the effective value
  in a bumped peer protocol so missing, mismatched, or old values fail before
  `ARMED` or data posting. Build deterministic aligned, non-overlapping,
  in-bounds first/middle/last extents, split every extent at slot boundaries,
  and make their lengths sum exactly to the effective budget. Full-payload CRC
  mode bypasses window sampling. Harden suffix parsing, multiplication,
  allocation, offsets, bounds, and overlap; free windows during cleanup. Test
  auto, minimum, non-multiple, exact one-MiB, boundary, over-capacity,
  overflow/allocation failure, determinism, protocol mismatch/mixed version,
  payload bypass, and sampled corruption in both receiver directions. The
  Python runner must relay only a complete protocol-v2 pair: validate every
  advertised endpoint, role, verification-mode, requested/effective budget,
  region/guard, transfer, queue, capacity and plan-consistency field before
  descriptor forwarding or data posting; reject v1, missing, extra, malformed
  or mismatched fields, including a PSN other than the binary's deterministic
  default. Reject result, CSV or done output before that negotiation completes,
  and require both endpoints, both matching verification messages, both result
  streams and successful done messages before returning success. Exercise that
  complete `run_bw.main`/relay path with local mocked endpoint processes.
- **Do not:** run a live benchmark/daemon, weaken the one-MiB gate, edit an
  authoritative checkout in place, replace an installed binary before all
  offline checks pass, touch networking/cables, use `SIGKILL`, or restart GLM.
- **Pass proof:** the exact one-QP/depth-one/four-MiB fixture creates windows
  totaling 1,048,576 bytes; impossible and mixed-protocol requests fail before
  data; both receiver directions detect sampled corruption; sanitizers and all
  focused, prior, and P06C0 tests pass; an end-to-end mocked runner test proves
  a complete matching v2 pair is forwarded while v1, incomplete, mismatched,
  command-inconsistent, premature-result, duplicate-done, and clean-but-
  incomplete pairs fail before producing a result file. Unterminated bytes at
  EOF and non-ASCII protocol bytes must also fail, including a duplicate done
  record without a final newline; correctly split ASCII records remain valid.
  Patched source and isolated Mac/Spark binary hashes are pinned; no process
  remains; GLM health/restart counts are unchanged. Only then may P06C retry
  Stage 1.

### P06C2 — Report receiver verification truthfully on both endpoint rows

- **Triggered by:** the second P06C Stage 1 run transferred all 12 pairs
  cleanly, but six READ responder rows reported `verified_bytes=0` because the
  receiver initiator could not return its proof in the `COMPLETE` message
- **Owner:** transport coding agent
- **Mode:** unattended, isolated patch/build/offline tests only
- **Scope:** the reproducible exact-verification patch, its manifest, focused
  protocol tests, and one receipt
- **Do:** version the completion message so a READ receiver sends its verified
  byte count, mismatch count, and guard result; validate that proof before the
  responder reports it. Preserve the existing WRITE receiver-to-initiator
  proof path and distinguish local receiver proof from peer receiver proof.
  Reject old, missing, duplicate, malformed, or contradictory proof fields.
- **Do not:** claim that a responder verified bytes locally when it did not,
  weaken the every-row one-MiB gate, contact a live host, replace an installed
  binary, change networking/cables, use `SIGKILL`, or restart GLM.
- **Pass proof:** both endpoint rows for every READ and WRITE trial truthfully
  report the receiver's exact 1,048,576-byte proof with zero mismatches and a
  successful guard; corruption and mixed/incomplete protocols fail closed;
  the full P06C1 suite and an independent adversarial review pass. P06C remains
  blocked until this child is Done.

### P06C — Run unattended MCDMA speed and recovery checks

- **Depends on:** P06A, P06B, P06B1, P06C0, P06C1, P06C2
- **Owner:** transport test agent
- **Mode:** unattended, bounded hardware tests
- **Scope:** throughput plus software-fault raw evidence and one receipt
- **Do:** run the pre-registered 12 throughput pairs and require exactly
  1,048,576 verified bytes in every measured row; only after that gate passes,
  run the no-service timeout, orderly daemon loss/fresh restart, and
  control-tunnel reconnect. Stop on the first hash, counter, generation,
  deadline, link, cleanup, OOM, restart, or API-health mismatch.
- **Do not:** pull a cable, change a route/interface/firewall/driver, use
  `SIGKILL`, restart GLM, change routing, or continue after failure.
- **Pass proof:** all endpoint exits, byte checks, counters, deadlines, and
  recovery checks pass; every measured row reports exactly 1,048,576 verified
  bytes; cleanup leaves nothing running; GLM health and restart counts match.

### P06D — Prove recovery from one owner-assisted cable disconnect

- **Depends on:** P06C and Spencer's explicit approval for the exact action
- **Owner:** transport test agent with Spencer
- **Mode:** owner-assisted, bounded hardware fault
- **Scope:** one ten-second disconnect and one receipt
- **Do:** after confirming cable identity, wait for the tested helper's exact
  safe prompt; Spencer disconnects only the verified Mac-facing QSFP28 on the
  head Spark for ten seconds and reconnects that same plug.
- **Do not:** move the inter-Spark or Thunderbolt cable, act on general consent,
  force recovery with settings, use `SIGKILL`, or restart GLM.
- **Pass proof:** interruption and counters are classified; link and a new
  daemon generation recover within fixed limits; a fresh transfer matches;
  cleanup passes and GLM stays healthy. P06 can then close.

### P07 — Choose the weight format and Mac layer boundary

- **Depends on:** P04
- **Owner:** model analysis agent
- **Mode:** unattended, read-only or isolated test
- **Scope:** decision D-010 successor plus one receipt
- **Do:** compare the MLX and EXL3 candidate layers numerically, confirm Spark
  disk headroom, prove the smallest candidate Mac block fits and runs alone,
  then choose one checkpoint policy and one boundary.
- **Pass proof:** the decision names exact revisions, layer interval, memory
  use, numerical difference, disk use, and a single selected policy.

### P07A — Run the complete selected Mac layer window

- **Triggered by:** P07 proved one quantized projection but found no installed
  GLM5 Next MLX implementation for the complete layer
- **Owner:** Metal model coding agent
- **Mode:** unattended, isolated model test only
- **Scope:** a pinned test-only GLM5 Next adapter/runtime under
  `experiments/p07a/`, its tests, `UPSTREAM.lock` if a source pin is added, and
  one receipt; do not edit the production pipeline yet
- **Do:** use the hash-matched extracted TensorFold 0.5.0 MLX implementation
  identified in the P07A source-map receipt; add only the minimum isolated
  adapter needed to run embedding plus layer `[0,1)` from MLX revision
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6`; capture deterministic inputs,
  all four residual streams, KDA state, output hash, memory, and repeated-run
  equality. Keep oMLX as a pinned reference, not the runtime.
- **Do not:** invent missing operations, use only a constituent projection as
  a pass, start a serving API, change the live Sparks, or download DFlash2.
- **Pass proof:** the complete Mac layer runs twice with identical output/state
  hashes, emits one BF16 `[rows,16384]` boundary buffer, remains within measured
  memory, and leaves no model process.

### P07A1 — Strengthen Mac tensor and offset proof

- **Triggered by:** independent P07A review found payload bytes unhashed and
  recorded-but-unasserted batch/chunk offsets
- **Owner:** Metal model coding agent
- **Mode:** unattended, isolated Mac fixture only
- **Scope:** P07A identity/fixture/tests/evidence plus one new receipt
- **Do:** hash the raw stored bytes for exactly the selected 53 tensors using
  safetensors offsets and record per-tensor plus combined digests; assert every
  batch/chunk offset match; rerun the two-process fixture.
- **Do not:** hash/load unrelated tensors, change/download the checkpoint,
  touch a Spark, invent a tolerance, start an API, or edit production code.
- **Pass proof:** 53 payload hashes and one combined digest reproduce across
  processes; every offset assertion passes; output/state behavior is unchanged
  or any difference is stopped and explained; no process remains and GLM is
  healthy. Only then may P07B begin after its separate P03 maintenance gate.

### P07B — Compare the complete layer on one Spark

- **Depends on:** P03, P07A, P07A1
- **Owner:** CUDA model integration agent
- **Mode:** owner-approved maintenance window
- **Scope:** isolated CUDA parity fixture plus one receipt; live recipe/runtime
  changes are prohibited
- **Do:** stage the same pinned MLX checkpoint on the head Spark, run the exact
  P07A token rows and complete layer `[0,1)` through the pinned TensorFold CUDA
  implementation, and compare every element of the post-layer BF16
  `[rows,16384]` buffer plus state identifiers.
- **Do not:** compare against the EXL3 weights, choose a tolerance before the
  result exists, alter the live model files, or leave a CUDA/model process.
- **Pass proof:** both sides used byte-identical checkpoint files and inputs;
  the receipt records maximum/mean/RMSE differences, output hashes, state
  checks, GPU/host memory, and cleanup. P07 becomes complete only after this
  observed result supports the selected policy and boundary.

### P08 — Freeze the bytes, state, timeouts, and acceptance limits

- **Depends on:** P01, P05, P07
- **Owner:** architecture agent
- **Mode:** unattended documentation
- **Scope:** `docs/TENSORFOLD-MCDMA-DESIGN.md`, `docs/DECISIONS.md`
- **Do:** freeze the four `[rows, 16384]` bf16 streams, 32 KiB per row,
  Mac-initiated request/reply flow, identifiers, checksums, deadlines, state
  owners, failure rules, shutdown rules, numerical tolerance, and maximum
  acceptable latency regression before integration tests begin. Put every
  field's type, byte order, range, and owner in one contract table. Define each
  timeout's start and stop event, every error and cleanup response, each
  numerical metric and limit, and the latency metric, P01 baseline runs,
  aggregation rule, allowed regression, and abort rule. Cite the measurement
  that supports each limit.
- **Do not:** choose or change a limit after seeing an integration result,
  change the P01 sample set, hide page-migration runs in an average, leave a
  value as TBD, or enable prefix reuse or MTP.
- **Pass proof:** a contract checklist reports zero missing or duplicate
  definitions; every field, owner, timeout, failure response, and test limit
  has one unambiguous value; the frozen contract hash and decision ID are
  recorded before P09 begins; prefix reuse and MTP are explicitly off.

### P09 — Add the versioned activation frame codec

- **Depends on:** P08
- **Owner:** coding agent
- **Mode:** unattended code work
- **Scope:** `src/tensorfold/pipeline/protocol.py` and its tests only
- **Do:** encode/decode the frozen frame and reject corrupt, stale, duplicate,
  oversized, truncated, wrong-generation, and timed-out input.
- **Pass proof:** `pytest -q tests/pipeline/test_protocol.py` passes and existing
  TensorFold tests remain unchanged.

### P10 — Add the TCP correctness transport

- **Depends on:** P09
- **Owner:** coding agent
- **Mode:** unattended code work
- **Scope:** `src/tensorfold/pipeline/tcp.py` and its tests only
- **Do:** carry the exact codec bytes through Mac-initiated request/reply with
  bounded queues, deadlines, clean disconnects, and no silent fallback.
- **Pass proof:** `pytest -q tests/pipeline/test_tcp.py` proves exact round trips
  plus timeout, disconnect, backpressure, and restart cases.

### P11 — Let CUDA load only its assigned layer interval

- **Depends on:** P00, P08
- **Owner:** CUDA coding agent
- **Mode:** unattended code work
- **Scope:** `families/glm5_next/cuda/weights.py` and targeted tests only
- **Do:** add validated start/end layer ownership while keeping the present
  full-model loader as the default.
- **Pass proof:** interval tests pass, invalid intervals fail closed, and the
  unchanged full loader produces the same manifest.

### P12 — Let CUDA continue from four hidden streams

- **Depends on:** P11
- **Owner:** CUDA coding agent
- **Mode:** unattended code work
- **Scope:** `families/glm5_next/cuda/forward.py` and targeted tests only
- **Do:** add a staged entry that accepts the exact validated four-stream frame
  at the chosen layer boundary without changing token-input behavior.
- **Pass proof:** targeted forward tests prove full-path equality on the local
  fixture and reject wrong shape, dtype, position, or interval.

### P13 — Mirror accepted frames to the second Spark

- **Depends on:** P12
- **Owner:** CUDA coding agent
- **Mode:** unattended code work
- **Scope:** `cuda/engine.py`, `cuda/decode.py`, `cuda/graphs.py`, and tests
- **Do:** have rank 0 validate one frame, mirror only accepted data to rank 1,
  and make calibration/graph capture start from the staged hidden input.
- **Pass proof:** both ranks report the same request, step, position, payload
  hash, and final fixture output; stale or rejected data reaches neither rank.

### P14 — Prove the full CUDA split over TCP

- **Depends on:** P03, P10, P13
- **Owner:** integration agent
- **Mode:** owner-approved maintenance window
- **Scope:** integration test and one receipt
- **Do:** run the chosen early interval on CUDA, send all four streams through
  the real codec and Mac relay over TCP, then finish on both Sparks.
- **Pass proof:** output is within the pre-set P08 limit versus the unsplit
  baseline, both ranks agree, and the live service is restored afterward.

### P15 — Add the MCDMA daemon transport

- **Depends on:** P05, P09
- **Owner:** transport coding agent
- **Mode:** unattended code work and isolated hardware test
- **Scope:** `src/tensorfold/pipeline/mcdma.py`, its tests, and one isolated
  hardware-test receipt only
- **Do:** first implement and test the daemon-owned RPC interface with a fake
  daemon, using the same bytes and state machine as TCP. After those tests pass,
  run one bounded real-daemon probe with the already accepted P05 configuration
  and record its transport identity and counter changes. TensorFold must not own
  RDMA queues directly.
- **Do not:** change an interface, driver, route, firewall, or cable; touch the
  live GLM containers; own RDMA queues directly; fall back to TCP; or leave a
  daemon, tunnel, listener, or client process running.
- **Pass proof:** `pytest -q tests/pipeline/test_mcdma.py` passes; the hardware
  receipt shows readiness and checksum probes succeeded before the request,
  the expected MCDMA counters moved, transport identity is MCDMA, no TCP
  connection was attempted, cleanup found no remaining process or listener,
  and GLM health and container restart counts are unchanged.

### P16 — Prove the full CUDA split over MCDMA

- **Depends on:** P03, P13, P15
- **Owner:** integration agent
- **Mode:** owner-approved maintenance window
- **Scope:** integration test and one receipt
- **Do:** repeat P14 with MCDMA as the only data transport.
- **Pass proof:** output meets the same P08 limit, both ranks agree, request
  counters prove MCDMA carried the frame, and the live service is restored.

### P17 — Load the Mac's assigned GLM layers

- **Depends on:** P07, P08
- **Owner:** Metal coding agent
- **Mode:** unattended code work
- **Scope:** new `families/glm5_next/mlx_stage.py` loader and tests only
- **Do:** load only the chosen embedding/early interval, validate its manifest,
  and expose no tokenizer, sampler, API, or second serving contract.
- **Pass proof:** loader tests match the P07 manifest and measured memory limit;
  missing, extra, or wrong-shaped weights fail closed.

### P18 — Run one deterministic Mac layer window

- **Depends on:** P09, P17
- **Owner:** Metal coding agent
- **Mode:** unattended code work
- **Scope:** `mlx_stage.py` forward path and tests only
- **Do:** implement `forward_window(rows)` for the chosen interval and emit the
  exact four-stream frame.
- **Pass proof:** fixed boundary fixtures meet the P08 numerical limit across
  repeated runs and reject wrong shape, dtype, position, or generation.

### P19 — Add Mac commit and reset state

- **Depends on:** P18
- **Owner:** Metal coding agent
- **Mode:** unattended code work
- **Scope:** `mlx_stage.py` state methods and tests only
- **Do:** add clean `reset` and a disabled-by-default `commit(keep)` foundation;
  ordinary decode may use reset but not speculative commit yet.
- **Pass proof:** reset returns to the exact initial-state hash, repeated calls
  do not leak state, and commit stays gated until P23.

### P20 — Prove synthetic three-machine GLM over TCP

- **Depends on:** P10, P13, P18, P19
- **Owner:** integration coding agent
- **Mode:** unattended isolated test
- **Scope:** TensorFold synthetic GLM fixture and integration tests only
- **Do:** extend the official fixture to four layers and run the Mac stage plus
  both Spark ranks through the real engine path over TCP.
- **Pass proof:** fixed seeds match the accepted unsplit fixture, both ranks
  participate, and errors fail closed.

### P21 — Prove synthetic MCDMA and three clean lifecycles

- **Depends on:** P15, P20
- **Owner:** integration agent
- **Mode:** unattended isolated hardware test
- **Scope:** integration tests and one receipt
- **Do:** repeat the synthetic run over MCDMA, then perform three clean
  start/request/stop cycles.
- **Pass proof:** TCP and MCDMA outputs match, counters identify every request,
  and all three cycles end with no orphan process or stale state.

### P22 — Prove one real GLM request across all three machines

- **Depends on:** P03, P16, P21
- **Owner:** integration agent with Spencer
- **Mode:** owner-approved maintenance window
- **Scope:** one bounded experiment receipt only
- **Do:** stop the live service and use MLX revision
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6` on all three machines. The Mac
  owns embedding plus base layer `[0, 1)` and hands off the post-layer-0
  four-stream residual buffer exactly as D-016 states. Keep prefix reuse and
  MTP off; compare fixed prompts, boundary hashes, top-1 choices, answer
  quality, and stop behavior. Stop if the checkpoint identity or interval
  differs.
- **Do not:** choose another checkpoint or interval, use EXL3 at the experiment
  boundary, enable prefix reuse or MTP, or continue after any identity or hash
  mismatch.
- **Pass proof:** all three machines and MCDMA counters appear in the trace,
  the receipt records the checkpoint hash, interval `[0, 1)`, and four boundary
  hashes, P08 limits pass, and the P02 rollback command immediately restores
  the original service and ends in `TWO_SPARK_BASELINE_OK`.

### P23 — Prove exact partial commit on the Mac stage

- **Depends on:** P19, P22
- **Owner:** Metal coding agent
- **Mode:** unattended code work, then isolated test
- **Scope:** Mac state path and focused tests only
- **Do:** for every row-window size `n` from two through eight, test every keep
  count `k` from zero through `n` against a fresh serial run with the same input
  and starting-state hash. Replay the discarded state exactly, then reset
  before the next case.
- **Do not:** sample only some keep counts, use a tolerance comparison, reuse
  mutated state across cases, enable MTP, or use a non-serial reference.
- **Pass proof:** all 42 `(n, k)` cases record equal output and state hashes plus
  the discarded-row replay hash; every reset returns the initial-state hash;
  and no model process remains.

### P24 — Enable and prove MTP as the only new variable

- **Depends on:** P23
- **Owner:** TensorFold integration agent
- **Mode:** owner-approved bounded model test
- **Scope:** `cuda/mtp.py`, coordinated state path, tests, and one receipt
- **Do:** copy the accepted P22 configuration, enable only its named checkpoint
  MTP setting, and record a normalized before/after configuration diff. Run the
  predeclared seeds, draft lengths, and every P23 keep case.
- **Do not:** change the checkpoint, boundary, transport, prompt, seed, timeout,
  or any other accepted setting; enable prefix reuse; make an unrecorded
  environment change; or continue if the configuration diff has another
  change.
- **Pass proof:** the normalized configuration diff contains exactly one MTP
  change; drafted and serial token, output, and state hashes match for every
  declared case; partial keeps are exact on all three machines; and the P02
  rollback command ends in `TWO_SPARK_BASELINE_OK`.

### P25 — Add joint prefix snapshots only if still wanted

- **Depends on:** P24
- **Owner:** state coding agent
- **Mode:** optional; owner-approved bounded model test
- **Scope:** Mac snapshot plus `cuda/decode.py` and focused tests
- **Do:** add one snapshot ID spanning Mac and both Sparks; never restore only
  one side.
- **Pass proof:** save/restore matches a no-reuse run by output and state hashes,
  and mismatched/missing IDs fail closed. Otherwise leave prefix reuse off.

### P26 — Run final correctness, speed, fault, and restart acceptance

- **Depends on:** P24; P25 only if prefix reuse is enabled
- **Owner:** acceptance agent with Spencer
- **Mode:** owner-approved maintenance windows
- **Scope:** receipts only
- **Do:** run five fixed seeds, long context, orderly daemon loss, cable loss,
  one Spark loss, corrupt frame, timeout, memory pressure, and restart tests.
  Before each fault, name the exact experiment process, confirmed cable, host,
  trigger, duration, abort limit, recovery command, and owner action. Reuse only
  P06D's confirmed Mac-facing cable. Test Spark loss by stopping only the
  experiment-owned rank process. Create memory pressure only inside an
  experiment process and cap it below P04's recorded safe free-memory limit.
- **Do not:** power off or reboot a host, stop a live GLM container, move an
  inter-Spark or Thunderbolt cable, use host-wide stress software, change a
  route, interface, firewall, or driver, use `SIGKILL`, restart a host, or act
  without Spencer's approval for that exact window and action.
- **Pass proof:** all pre-set P08 correctness/performance limits pass, every
  failure matches the fail-closed rule, and three clean lifecycles pass. Each
  fault receipt records pre/post service health, container restart counts,
  exact PID or cable identity, timestamps, bounded failure code, and a clean
  process/listener scan. After each window the P02 rollback command restores
  the original service and ends in `TWO_SPARK_BASELINE_OK`.

### P27 — Reproduce cleanly and prepare upstream changes

- **Depends on:** P26
- **Owner:** release agent
- **Mode:** unattended until publication approval
- **Scope:** pinned setup, sanitized tests/docs/receipts, and small upstream
  change sets
- **Do:** record the approved disposable rehearsal host or environment and its
  pinned base identity before work. Reproduce from a clean checkout using only
  the documented commit and lockfiles, separate generic TensorFold/MCDMA work
  from private host configuration, prepare minimal local upstream branches,
  and name the secret scanner, scanned paths, exclusions, and zero-finding
  rule.
- **Do not:** install on a production host, copy credentials or private receipts
  into an upstream branch, delete pre-existing host data, push a branch,
  publish anything, or open a pull request without Spencer's approval.
- **Pass proof:** the rehearsal receipt records the clean base identity,
  commit and lockfile hashes, exact commands and exit codes, and artifact
  hashes; the declared scan has zero findings; a file manifest proves generic
  and private files are separated; and no branch or pull request was published.

### OPT1 — Evaluate OpenResearch as a results browser

- **Depends on:** P01 only; never blocks P02-P27
- **Owner:** tooling agent
- **Mode:** optional, control Mac only
- **Scope:** one evaluation receipt; no live-host installation
- **Do:** create and record a new trial directory on the control Mac, install
  only inside it, bind services to loopback, and use copied non-secret sample
  receipts to test whether OpenResearch improves experiment browsing beyond
  Git, Linear, receipts, and the append-only experiment log. Keep telemetry
  off. If rejected, remove only the recorded trial files and processes.
- **Do not:** touch an existing OpenResearch installation or profile, install
  on a live host, use credentials or private receipts, enable telemetry, expose
  a non-loopback listener, or delete a path not listed in the trial inventory.
- **Pass proof:** the receipt records setup minutes, duplicate-artifact count,
  evidence-quality rubric score, cleanup result, and a written keep/reject
  decision. The before/after inventory accounts for every trial-owned change;
  telemetry and non-loopback listener checks are empty; and a rejected trial
  leaves no trial process, listener, or file behind.
- **Current evidence:** the sanitized
  [Windows controller receipt](../results/receipts/OPT1-windows-openresearch-controller-2026-10-01.md)
  records the verified controller identity, canary, health, shutdown, cleanup,
  and no-change facts. It does not claim a model optimization ran.

### OPT2 — Use a fixed offline autoresearch evaluator (MOT-3370)

- **Depends on:** none; never blocks P00-P27
- **Owner:** tooling agent
- **Mode:** offline only
- **Scope:** `experiments/autoresearch/`, `docs/AUTORESEARCH.md`, and immutable
  local receipts
- **Do:** freeze a baseline and evaluator, allow exactly one setting change per
  trial, check correct answers and safety before speed, and record an explicit
  keep/reject result plus whether work must stop. Use OpenResearch 0.2.14 on
  the Windows desktop only as an optional controller and record browser.
- **Do not:** run Karpathy's nanochat trainer, require OpenResearch at runtime,
  contact a model host, call a network service, change the live model, or treat
  `CONTROLLER_READY` as proof that an optimization ran.
- **Pass proof:** every required rejection fixture passes offline, repeated
  identical input creates the same content-addressed receipt, no test contacts
  a host, and the receipt clearly says keep or reject and stop or continue.

## Completion rule

The parent issue may close only after P00-P24, P26, and P27 pass. P25 is needed
only if prefix reuse is enabled. OPT1 and OPT2 are never required for model
completion.
