# Three-machine TensorFold and MCDMA completion plan

> Status correction, 2026-10-03: the prior sixteen-request capacity win is not
> completion of the requested single-request design. The active acceptance
> contract is `GOAL-SINGLE-STREAM-MCDMA.md`.

## Objective

Run one GLM-5.3 Flash inference path where both Sparks prefill one prompt,
MCDMA moves its complete live cache, and the Mac Studio decodes the same reply.
It must beat the complete request time implied by MiaAI-Lab v1.5's published
two-Spark TensorFold TTFT and decode-rate claims by at least 3%, using the same
prompts and greedy settings. Exact output is checked against the retained
same-checkpoint control; useful context, reproducible measurements, recovery,
and clean shutdown remain mandatory.

The previous two-Spark service is preserved on disk but is not the objective or
the completion standard.

## Definition of done

1. Both Sparks execute real prompt prefill as a two-rank TensorFold CUDA group.
2. MCDMA moves the complete prompt cache and its counters advance without
   failures.
3. The Mac imports that exact state and performs real TensorFold Metal decode.
4. Exact token IDs match the same-model two-Spark reference.
5. Three clean stop/start/answer cycles pass without orphaned owned processes.
6. Startup pins and checks every image, model, stage file, runtime, and library.
7. A repeated one-request benchmark beats MiaAI-Lab v1.5's published C1 target
   by at least 3% end to end and records the complete timing split,
   context/cache behavior, output hash, clocks, and rejected tuning attempts.
8. Unit and lifecycle tests pass while the live service remains online.
9. A Fable Ultra adversarial review finds no unresolved completion blocker.
10. Sanitized runbook, design, code map, decisions, failures, and evidence are
    committed and suitable for an upstream review.

The first accepted release is a single-request greedy text service. Streaming,
concurrency, tools, vision, and full Mia API parity remain later features. MTP
or another proven drafter is part of the performance comparison when the
two-Spark baseline uses it.

## Current state

| Gate | State | Evidence |
| --- | --- | --- |
| Hardware and MCDMA data path | pass | replacement card enumerated; checked transfers and live model calls pass |
| Compatible source and artifacts | pass | Mia v1.5 and sparkDash claims pinned as the external target; official TensorFold 0.6.5 staged and hash-checked on the Mac; candidate image/model pins retained |
| Fixed frame and transaction rules | pass | protocol, mailbox, and transaction unit tests |
| Real Mac TensorFold stage | pass | embedding plus original layer 0 on Metal |
| Real dual-Spark TensorFold stage | pass | original layers 1..44, final norm/head, rank agreement |
| End-to-end GLM over MCDMA | pass | exact count sequence and exact readiness response |
| Repeatable lifecycle | pass | three clean stop/start/answer cycles after recorded repairs |
| Persistent API | pass | warm exact request and health endpoint |
| Mac eight-stream capacity | pass | 84.9-87.9 aggregate tok/s; 48/48 exact; slowest first token 1.66 s |
| Separate sixteen-request capacity result | pass, not acceptance | hybrid 172.97-173.09 aggregate tok/s vs two-Spark 137.48-137.80; this is concurrency, not the requested one-request path |
| Spark-prefill / MCDMA / Mac-decode single request | fail, active work | earlier four-request form was 56.72% slower and supported only roughly 2K total tokens; a current one-request campaign is still required |
| Current test suite | pass | Python and shell/lifecycle suites |
| Sanitized documentation | pass | runbook, design, experiment log, and content-addressed receipts retained |
| Fable Ultra adversarial review | pass with recorded repairs | REVIEW-012 completed; all supported P1/P2 findings repaired and tested; REVIEW-013 weekly-limit retry is not counted as approval |
| Clean tracked release state | pass | final checks and cleanup complete on `codex/three-machine-inference` |

## Implemented architecture

```text
Mac API/tokenizer
  -> TensorFold Metal embedding + layer 0
  -> MCDMA BF16 activation
  -> head Spark TensorFold CUDA rank 0
  <-> worker Spark TensorFold CUDA rank 1 over NCCL
  -> layers 1..44 + norm/head + greedy token
  -> MCDMA token/control reply
  -> Mac acknowledgement commits all three caches
```

## Completion state

The transport, correctness proof, lifecycle work, and separate concurrency
campaign are retained. The requested single-request performance goal is open.
No service is described as complete until the scorecard in
`GOAL-SINGLE-STREAM-MCDMA.md` passes twice from clean starts.

## Measurement policy

- one setting changes per experiment;
- every run uses a fixed prompt and token count;
- greedy output hashes must remain identical;
- raw logs stay ignored and receipts record SHA-256 hashes;
- rejected experiments remain in the record;
- performance is reported, never inferred from negotiated link speed; and
- correctness, transport health, lifecycle recovery, and speed are separate
  gates.

## Stop conditions

Stop and repair before proceeding if any rank disagrees, a model/artifact hash
changes, MCDMA records a failure, output changes unexpectedly, a rank is OOM
killed, shutdown leaves owned children, the API silently falls back, or the
review finds an unresolved correctness or recovery blocker.

## Upstream package

The upstream-ready package includes the generic protocol, mailbox, transaction,
Metal stage, partial CUDA runner, lifecycle scripts, tests, version pins,
benchmark receipt, negative results, and known limits. It excludes private
host configuration, credentials, addresses, raw logs, model weights, and the
non-commercial DFlash2 add-on.
