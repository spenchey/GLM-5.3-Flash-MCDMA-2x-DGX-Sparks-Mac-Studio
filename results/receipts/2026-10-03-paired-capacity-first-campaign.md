# First same-width paired-capacity campaign — 2026-10-03

## Result

**REJECTED.** This campaign used commit
`17a0536d763368cbc221b39e7145dc97dd8ed83b`, TensorFold `0.6.3`, one fresh
four-lane two-Spark service for arm A, and eight Spark plus eight Mac lanes for
arm B. Both arms handled sixteen identical 256-token requests. The schedule was
one warmup per arm followed by `ABBABAAB`.

The baseline median was `137.731` tokens/s. The candidate median was `168.147`
tokens/s. Three of four pairs beat the baseline by `17.7%` to `27.0%`, but the
second candidate round fell to `17.437` tokens/s and `234.798` seconds. The
other candidate rounds were `174.114`, `174.007`, and `162.287` tokens/s. The
single collapse makes the campaign fail its every-pair, tail-latency, and
confidence-bound gates.

All outputs retained the frozen token and byte hashes. Every candidate round
used exactly five MCDMA calls and three frames. Every baseline round used zero
MCDMA calls. Container identities and configurations stayed fixed, no OOM or
restart was observed, MCDMA reported no failure, and cleanup left no owned
container or Mac worker running.

## Discriminating evidence

Only the eight Mac lanes stalled in the rejected round; all eight finished at
the same `234.798` seconds. The Spark lanes stayed in their normal range. The
next two-Spark round and the following candidate round returned to normal. The
Studio reported no swap, thermal warning, CPU power warning, or memory
shortage. This isolates the problem to transient Mac TensorFold/MLX round
lifecycle behavior rather than correctness, the Spark service, or the MCDMA
link.

Static inspection then found two exact deviations from TensorFold `0.6.3`'s
official Mac path:

1. the worker called the internal GLM runtime loader, bypassing the public
   loader's Metal wired-memory limit; and
2. each temporary lane engine retained its completed round state and freed MLX
   buffers were not reclaimed before the next measured round.

Those deviations are the next controlled repair, not yet a proven explanation.
The same unchanged paired campaign must pass twice before the repair is
accepted.

## Evidence

- raw directory:
  `results/raw/paired-capacity-p16-s8-m8-t256-20261003t065034z-59565/`
- `acceptance.json` SHA-256:
  `886c7334b61a920712b74c929d476a67a73ac68640c3277554fcd582df3fdbf5`
- `rounds.jsonl` SHA-256:
  `e45beb2d6914f13e9a44bff5c22f4f16ed8e839399d9f57ea7c376c255f473b1`
- `worker.stderr` SHA-256:
  `aab226b1abee3a1d3fede05b1754206ce9a6fbc422a6a600adc4e5ad1720bd3b`

