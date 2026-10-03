# Three-machine TensorFold and MCDMA completion plan

## Objective

Run one GLM-5.3 Flash inference path across the Mac Studio and both DGX Sparks:
TensorFold Metal on the Mac, TensorFold CUDA on both Sparks, and real MCDMA for
the Mac-to-Spark model boundary. The result must be reproducible, recoverable,
measured, fail closed, documented, and ready for an upstream technical review.

The previous two-Spark service is preserved on disk but is not the objective or
the completion standard.

## Definition of done

1. The Mac executes real GLM compute, not a relay or synthetic fixture.
2. Both Sparks execute the remaining real GLM layers as a two-rank TensorFold
   CUDA group.
3. Real model activations cross MCDMA and its counters move without failures.
4. Exact semantic answers and deterministic repeated outputs pass.
5. Three clean stop/start/answer cycles pass without orphaned owned processes.
6. Startup pins and checks every image, model, stage file, runtime, and library.
7. A repeated benchmark records speed, timing split, output hash, and rejected
   tuning attempts.
8. Unit and lifecycle tests pass while the live service remains online.
9. A Fable Ultra adversarial review finds no unresolved completion blocker.
10. Sanitized runbook, design, code map, decisions, failures, and evidence are
    committed and suitable for an upstream review.

The first accepted release is a single-request greedy text service. MTP, prefix
reuse, streaming, concurrency, tools, vision, and feature parity with Mia's
complete two-Spark server are later features, not hidden completion criteria.

## Current state

| Gate | State | Evidence |
| --- | --- | --- |
| Hardware and MCDMA data path | pass | replacement card enumerated; checked transfers and live model calls pass |
| Compatible source and artifacts | pass | Mia `main` at `cf28cc4` on Sparks / official TensorFold 0.6.3 on Mac / pinned model and image checks |
| Fixed frame and transaction rules | pass | protocol, mailbox, and transaction unit tests |
| Real Mac TensorFold stage | pass | embedding plus original layer 0 on Metal |
| Real dual-Spark TensorFold stage | pass | original layers 1..44, final norm/head, rank agreement |
| End-to-end GLM over MCDMA | pass | exact count sequence and exact readiness response |
| Repeatable lifecycle | pass | three clean stop/start/answer cycles after recorded repairs |
| Persistent API | pass | warm exact request and health endpoint |
| Mac eight-stream capacity | pass | 84.9-87.9 aggregate tok/s; 48/48 exact; slowest first token 1.66 s |
| Fair same-width capacity win | pass | two clean 16-request campaigns: hybrid 172.97-173.09 tok/s vs two-Spark 137.48-137.80; every pair exact and accepted |
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

The implementation, adversarial review, two clean performance campaigns,
repeatability comparison, documentation, full test suite, and clean shutdown
are complete. The benchmark service is deliberately stopped between campaigns;
publication, production routing, and an upstream pull request remain separate
owner-approved actions.

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
