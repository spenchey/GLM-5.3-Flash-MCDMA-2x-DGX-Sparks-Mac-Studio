# Changelog

## 2026-10-04

- Added protocol 4 segment proofs so final cache sealing no longer rereads the
  complete 148 MB transfer. The retained five-request median improved from
  6.999 to 6.912 seconds while preserving zero MCDMA failures.
- Measured MCDMA Spark-push against Mac-pull with the same model, prompt, cache,
  frame shape, and decode path. Push shortened the median handoff by about 3 ms
  but did not improve complete time, so the retained configuration stays on
  pull.
- Preserved exact canary and five-request evidence, restored the accepted local
  setting, and verified clean shutdown of every owned process.

## 2026-10-03

- Added and measured a GLM-compatible Mac DFlash2 path without downloading
  MiaAI-Lab's base-model checkpoint. It raised median Mac decode to 61.28
  tokens/s, but the complete cold three-machine request remained slower than
  Mia's published two-Spark result because the cache handoff added about 0.52 s.
- Rebased the checked Mac TensorFold stage from 0.6.4 to the newly released
  0.6.5; GLM family files and the dense-stage patch are unchanged, while the
  complete source and patched trees have new verified hashes.
- Certified two repeatable sixteen-request campaigns in which the three-machine
  path reached 172.97-173.09 aggregate tokens/s, 25.7-25.9% above the matched
  two-Spark baseline, with exact output in every lane.
- Added the Apache-2.0 license, third-party NOTICE, complete credits chain,
  security guidance, and a public-release check that rejects private addresses,
  home paths, credential patterns, and tracked model artifacts.
- Split the public recipe from proposed upstream work: one clean MCDMA cleanup
  change can be submitted after its upstream suite passes; the TensorFold
  loader change must first be measured and packaged for TensorFold 0.6.5.

## 2026-10-02

- Implemented real three-machine GLM-5.3 Flash inference: TensorFold Metal
  embedding/layer 0 on the Mac, TensorFold CUDA layers 1..44 on both Sparks,
  and real MCDMA activations between them.
- Added a checksum-protected, deadline-bound frame protocol and two-phase cache
  commit so Metal and both CUDA ranks advance together.
- Added a persistent Mac OpenAI-compatible endpoint and exact semantic canaries.
- Passed three clean stop/start/answer cycles after preserving and repairing
  three packaging/status-reader failures.
- Added deterministic five-run performance measurement, rejected a slower L2
  prefetch candidate, and restored the accepted configuration.
- Added pinned preflight, deployment, status, chat, benchmark, and orderly stop
  scripts plus protocol, mailbox, transaction, stage-patch, and lifecycle tests.
- Added fail-closed partial-rank recovery that preserves evidence, verifies the
  exact owned API and containers before signaling, restarts all three machines,
  and requires an exact inference canary.
- Replaced stale planning documents with the implemented architecture, operator
  runbook, sanitized receipts, known limits, and upstream contribution map.

## 2026-10-01

- Established the pinned, healthy two-Spark TensorFold GLM baseline.
- Verified the separate Studio/head-Spark MCDMA transport.
- Documented that TensorFold does not yet use MCDMA.
- Locked TensorFold as the final serving engine and oMLX as reference code.
- Replaced the one-Spark intermediate model milestone with a direct
  Mac-plus-two-Sparks plan.
- Added decision, experiment, failure, compatibility, and completion records.
- Added a raw experiment runner that timestamps output and emits its SHA-256.
- Fixed the worker status command so a healthy worker is no longer followed by
  a false `running=false` line.
- Completed a read-only Claude Fable maximum-effort adversarial review and
  preserved its model, session, prompt, and raw-output hashes.
- Replaced the impossible parallel full-model canary with a rehearsed
  maintenance-window stop/test/restore cycle.
- Added pre-code gates for the live patched TensorFold source, unchanged
  benchmarks, `mcdma-rpcd` hardware proof, and checkpoint-policy evidence.
- Corrected the GLM boundary to four `[rows, 16384]` bf16 residual streams and
  the transport direction to Mac-initiated request/reply.
- Added CUDA loopback, partial-commit, prefix-state, daemon placement, security,
  and pre-registered performance-bound requirements.
- Refreshed the research checkout to TensorFold 0.6.1 and staged Mia recipe
  v1.3.2 with its supported TensorFold 0.6.0 image on both Sparks, while
  preserving the healthy 0.5.0 service as the rollback baseline.
