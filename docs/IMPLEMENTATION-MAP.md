# Three-machine implementation map

## Runtime code

| File | Responsibility |
| --- | --- |
| `experiments/three_machine/protocol.py` | Fixed frame format, checksum, deadlines, shape rules, token and control frames |
| `experiments/three_machine/mailbox.py` | MCDMA client/service mailbox bindings |
| `experiments/three_machine/transaction.py` | One-stream ordering, exact retry, pending token, two-phase commit |
| `experiments/three_machine/mac_stage.py` | TensorFold Metal embedding, original layer 0, local cache, tokenize/decode loop |
| `experiments/three_machine/cuda_partial.py` | TensorFold CUDA original layers 1..44, NCCL rank agreement, norm/head/token |
| `experiments/three_machine/api_server.py` | Persistent Metal stage and small OpenAI-compatible API |
| `experiments/three_machine/control.py` | Reset and orderly model shutdown |
| `experiments/three_machine/export_stage_checkpoint.py` | Deterministic minimal Mac stage export |
| `experiments/three_machine/frame_echo.py` | Framed transport diagnostic |
| `experiments/three_machine/cache_handoff.py` | Authenticated multi-tensor prompt-cache manifest and frames |
| `experiments/three_machine/cuda_prefill.py` | Export one immutable portable-checkpoint prefix cache on the Spark pair |
| `experiments/three_machine/mac_decode.py` | Import one verified cache and decode up to eight shared Metal lanes |
| `experiments/three_machine/concurrent_capacity.py` | One-clock, byte-exact paired-capacity worker |
| `experiments/three_machine/native_attestation.py` | Per-round full baseline container/restart/health attestation |
| `experiments/three_machine/paired_capacity.py` | Fail-closed paired performance acceptance evaluator |
| `experiments/three_machine/paired_repeatability.py` | Require two independently recreated accepted campaigns with the same frozen inputs and configuration |

## TensorFold change

`patches/tensorfold/0001-metal-read-exl3-dense-stage.patch` adds the narrow
Metal-side ability to read the required EXL3 dense tensors for embedding and
original layer 0. It does not replace TensorFold's model implementation.
`patches/tensorfold/0002-glm-dflash2-metal.patch` adds the reviewed Mac GLM
DFlash2 loader and verified draft-head runtime. It remains optional at launch.
`patches/tensorfold/metal-stage-manifest.json` pins both exact input and output
source hashes. `scripts/prepare-tensorfold-metal-stage.sh` refuses any base
other than official TensorFold 0.6.5 at the pinned commit, applies both patches
in a new destination, and verifies the complete result.

The deployed Spark image is Mia's pinned TensorFold 0.6.0 image with its exact
published patch label. Both ranks verify the same image ID before startup. The
Mac uses official TensorFold 0.6.5 in an isolated source stage with the checked
dense-stage and GLM DFlash2 patches. The asymmetric versions are intentional
and independently pinned in `UPSTREAM.lock`.

## Operations

| Script | Responsibility |
| --- | --- |
| `scripts/preflight-three-machine.sh` | Verify every pinned image, model, stage, runtime and library |
| `scripts/prepare-tensorfold-metal-stage.sh` | Materialize the exact checked Mac TensorFold source from the pinned official base |
| `scripts/deploy-three-machine.sh` | Copy the controlled project subset to all runtime hosts |
| `scripts/project-manifest.py` | Prove the reusable deployed file set is byte-identical on the controller and all three hosts |
| `scripts/start-three-machine.sh` | Start MCDMA, both ranks, and the Mac API in dependency order |
| `scripts/status-three-machine.sh` | Report both containers, MCDMA state/counters, and API health |
| `scripts/chat-three-machine.sh` | Send one verified greedy chat request |
| `scripts/benchmark-three-machine.py` | Repeat one fixed request and enforce one output hash |
| `scripts/benchmark-three-machine.sh` | Run that benchmark against the Mac API |
| `scripts/stop-three-machine.sh` | Orderly model shutdown followed by MCDMA cleanup |
| `scripts/recover-three-machine.sh` | Preserve evidence, stop exact failed owners, restart all three machines, and require an exact canary |
| `scripts/prepare-concurrent-cache.sh` | Freeze the shared prefix, exact 256-token reply, cache bytes/frames/transfer identity, and component preparation timings |
| `scripts/benchmark-concurrent-capacity.sh` | Compare fresh two-Spark and 8+8 hybrid rounds without reloading either engine |
| `scripts/stop-concurrent-worker.py` | Stop only the exact owned remote paired-capacity worker |

`scripts/tests/p06-live-adapter.py` is the bounded MCDMA process owner used by
the lifecycle scripts. Its `peek` operation is read-only. Its control reader
collects complete multi-line daemon status replies rather than assuming one
socket read contains the full reply.

## Test map

| Test | Proof |
| --- | --- |
| `tests/test_three_machine_protocol.py` | Header, payload, checksum, deadline, shape and control rejection |
| `tests/test_three_machine_transaction.py` | Ordering, duplicate replay, acknowledgement and reset rules |
| `tests/test_three_machine_mailbox.py` | Client/service MCDMA mailbox behavior |
| `tests/test_concurrent_capacity.py` | Eight-lane worker timing, exact-output, TensorFold identity, and owned-process protocol behavior |
| `tests/test_native_attestation.py` | Head, worker, and replay container identity, configuration, restart, health, and role checks |
| `tests/test_paired_capacity.py` | Fixed paired schedule, exact Spark/Mac lane topology, cache/MCDMA binding, exact output, raw-lane-derived throughput/tail metrics, and continuity gates |
| `tests/test_paired_repeatability.py` | Two clean campaigns must recreate all containers while retaining identical frozen inputs and stable configuration |
| `tests/test_tensorfold_stage_patch.py` | Stage patch/manifest hashes and required narrow reader hunks |
| `tests/test-p06-live-adapter.py` | Bounded process ownership, fragmented status reply, reusable clean directory |
| `tests/test-mcdma-control-tunnel.sh` | Private tunnel, conflict refusal, isolation and clean stop on a dedicated test port |

The live evidence is separate from unit tests. It is indexed in
`docs/EXPERIMENT-LOG.md` and summarized by the three 2026-10-02 receipts under
`results/receipts/`.

## Deliberately absent from the original layer-split service

- no TCP fallback in the accepted service;
- no MTP or DFlash2 in the original correctness service;
- no prefix-cache snapshots in the original correctness service;
- no sampled decoding;
- no request concurrency or streaming; and
- no claim that this small API equals the full Mia server surface.

The capacity candidate now implements the narrow shared-prefix cache and uses
DFlash2 only in the named proof-of-concept baseline. Arbitrary per-request
prefix export, full API parity, and production licensing remain follow-up work.
