# Progressive MCDMA frame-size sweep

Date: 2026-10-04

## Question

Can a larger progressive MCDMA frame remove enough fixed call overhead to make
the cold two-Spark-prefill/Mac-decode path beat MiaAI-Lab's published C1 result
by 3%?

## Controlled sweep

All three canaries used protocol 3, the same 33-token prompt, the same
148,032,512-byte real Spark cache, 16 forced greedy output tokens, and the same
deployed project commit `6699d19`. Every result matched the Mac-local reference
exactly and MCDMA reported zero failures.

| Requested limit | Effective frames | Transfer | Handoff total | Result |
| ---: | ---: | ---: | ---: | --- |
| 64 MiB | 3 | 0.498972 s | 0.496563 s | valid |
| 96 MiB | 2 | 0.475359 s | 0.472194 s | valid; selected |
| 160 MiB | 1 | 0.482267 s | 0.479144 s | runtime valid; analyzer false negative |

The 160 MiB runtime result is valid: its single frame carried the complete
cache and the output matched. The original analyzer incorrectly required the
effective payload capacity to equal the nominal mailbox size. MCDMA reserves
69,640 bytes for its reply envelope, so the effective capacity was
167,702,520 bytes rather than 167,772,160. The canary now accepts a positive
effective capacity no larger than the requested limit and checks that the
resulting frame count covers the complete cache.

## Five-request 96 MiB comparison

The selected two-frame shape was then run through the frozen five-request cold
benchmark.

- Median complete time: `6.999304583` s.
- Required complete time: `6.572681457` s or less.
- Remaining gap: `0.426623126` s.
- Median TTFT: `0.481589375` s.
- Maximum complete time: `9.573404792` s; the first request paid CUDA
  compilation.
- Median Mac decode: `61.200419` tokens/s.
- First local-reference difference: token `104` on every run.
- Protocol: `3`, progressive execution-order frames, positively attested
  DFlash.
- MCDMA failures and OOM signals: zero.
- Cleanup: both Spark ranks, the Mac connector, and the MCDMA link stopped.

The steady handoffs after the compilation request were `0.467249` to
`0.474579` seconds. Their first 96 MiB frame calls were `0.261547` to
`0.274257` seconds; second-frame calls were `0.037001` to `0.041051` seconds.
Assembly remained about 57-62 ms and final verification about 49-54 ms.

## Evidence

- 64 MiB result:
  `results/raw/progressive-cache-canary-64m-20261004T060043Z-14049/result.json`,
  SHA-256 `9d0af32437330254d09f1e2f12f2bea99ea9801df21159213c32c0f60b2c1a35`.
- 96 MiB result:
  `results/raw/progressive-cache-canary-96m-20261004T060157Z-14467/result.json`,
  SHA-256 `20c6be26694efb551a2f7d3118ae1bf5830754bc4b140d7a2bb7f16033232644`.
- 160 MiB result:
  `results/raw/progressive-cache-canary-160m-20261004T060310Z-14915/result.json`,
  SHA-256 `c24422745f53cc0bd670e0e243a48ffe59435bbaef1db3ae97aa558101e318e7`.
- Five-request benchmark:
  `results/raw/mia-c1-three-machine-cold-dflash-progressive-96m-20261004T061048Z-17690/`.
- Benchmark result SHA-256:
  `fea25709d9fc6106271b79e1746b01fd6c0e39c5ff9a08d1bfafae78faed70e0`.
- Benchmark summary SHA-256:
  `32ca82846b0d4db067faf265b7375524c6cd9939e5a89ad7cffaff9c1f63e18a`.

## Decision

Keep 96 MiB as the measured progressive frame limit, but close frame-size
tuning. It improved the 32 MiB median by about 33 ms and cannot close the
remaining 427 ms. The next candidate must remove producer/consumer copies or
hash passes and improve the Mac target forward. Exactness at token 104 and
startup compilation remain independent acceptance failures.
