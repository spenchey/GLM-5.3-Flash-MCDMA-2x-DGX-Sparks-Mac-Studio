# Protocol 4 integrity reuse and MCDMA reply-direction comparison

Date: 2026-10-04

## Scope and pins

- Project commit: `5720d0b`.
- TensorFold base: `609ca419abecebdc5a059498a613680bd3aa847f`
  (0.6.5), with the reviewed GLM Mac DFlash stage.
- TensorFold source manifest:
  `13cf04f5f4975d9f9517f106d4ae993f9fbac0fe1191fb62272c79e80f92b337`.
- MCDMA source pin: `e672c14ff9fc7b38994caf73025cf1588b4de74e`.
- Spark image:
  `sha256:446a23697c7eba9e0434cb162b5389792c7a5ab0f2652173bf1e791e7fdb4c5d`.
- Model: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` snapshot
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6` on all three machines.
- Frozen workload: MiaAI-Lab C1 prose prompt, greedy generation, thinking off,
  400 forced output tokens, one request at a time, DFlash, two progressive
  frames with a 96 MiB requested limit.

## Protocol 4 result

Protocol 4 authenticates each tensor segment while the receiver copies it and
reuses those proofs at the final seal. The Spark reuses the digest it computed
when each immutable tensor became final. This removes the second full
148,032,512-byte integrity read while retaining corruption rejection.

The complete 292-test suite passed. A warm live canary using the retained
Mac-pull transport reported:

- exact Spark-prefill versus Mac-prefill output;
- `0.377630417` seconds observed transfer;
- `0.375309375` seconds measured handoff total;
- `0.252530042` seconds in the two MCDMA frame calls;
- `0.000777417` seconds final verification;
- zero MCDMA failures.

The matching five-request cold benchmark reached:

- `6.912238584` seconds median complete time;
- `0.397171500` seconds median TTFT;
- `61.118118` median decode tokens/s;
- `6.945360500` seconds maximum complete time;
- zero MCDMA failures and zero Spark OOM signals.

This improved the prior protocol-3 median by `0.087066` seconds, but still
missed the required `6.572681457`-second ceiling by `0.339557127` seconds and
still differed from the Mac-local control first at token 104.

## Push versus pull

MCDMA supports two reply directions. The retained setup has the Mac pull each
completed frame with RDMA READ. The one intentional change in the comparison
was to let the Spark write each reply directly into the Mac mailbox. Before the
test, both live Studio ports reported `MCDMARelaxedOrdering = No`, the safety
condition named by MCDMA for ordered reply writes.

The warm push canary was exact and reduced observed transfer to
`0.358208125` seconds. Its two frame calls took `0.232685585` seconds, about
20 ms below the adjacent pull canary. The full five-request result did not
retain that improvement:

| Measure | Pull | Push |
| --- | ---: | ---: |
| Median complete | 6.912239 s | 6.917320 s |
| Median TTFT | 0.397171 s | 0.396361 s |
| Maximum complete | 6.945360 s | 6.928586 s |
| Median decode | 61.118 tok/s | 61.186 tok/s |
| Median handoff | 0.390569 s | 0.387597 s |
| Exact-output first difference | token 104 | token 104 |
| MCDMA failures / Spark OOMs | 0 / 0 | 0 / 0 |

The approximately 3 ms median handoff reduction was lost inside normal decode
variation and complete time was 5 ms slower. Spark-push is therefore rejected
for the frozen single-stream goal. The ignored local configuration was restored
to Mac-pull after the run.

## Evidence

- Pull canary result:
  `results/raw/progressive-cache-canary-96m-20261004T064402Z-29214/result.json`,
  SHA-256 `9753e3ff5e0e0a71a08751d474bd4ee8b8c3f0babf475c5f2adb14520896cc39`.
- Pull five-request result:
  `results/raw/mia-c1-three-machine-cold-dflash-progressive-96m-20261004T064525Z-29705/result.json`,
  SHA-256 `d96ff825ea7e9ccb668e96c2c3e6a3b0cd16599eb46f3424deb44423b98b41d6`.
- Pull summary SHA-256:
  `6acf31db2f53f9d6aa95443bb3e44e5a3d9e8b105b5ddcc512cf48689e668dff`.
- Push canary result:
  `results/raw/progressive-cache-canary-96m-20261004T070251Z-36300/result.json`,
  SHA-256 `bcd299d03164d6f655a8d71bca74319ece331ce192c6eda64fee876633962a0e`.
- Push five-request result:
  `results/raw/mia-c1-three-machine-cold-dflash-progressive-96m-20261004T070414Z-36905/result.json`,
  SHA-256 `36e56416cabf919f70cadf84a197f7978f648efb014c17cda295bb9005c3bec5`.
- Push summary SHA-256:
  `60f6dacd85e8b223394195eceaca9aaa93bdfa5e3bb1c7962adcd304b7ab60f5`.
- Push cleanup stdout SHA-256:
  `f1a434ff03d53ca11326a016be55d5babf26eaba4b42b4aff6e837a0fcb0a5c2`;
  cleanup stderr was empty.

## Decision

Keep protocol 4 and the 96 MiB two-frame shape. Keep Mac-pull for the retained
configuration. Reply direction is closed as a useful single-stream tuning
axis; the remaining work is Mac target-forward performance, token-104
cross-backend equality, and complete long-context state transfer.
