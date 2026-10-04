# Current Mia v1.5 Spark image candidate

Date: 2026-10-04

## Question

Does MiaAI-Lab's current v1.5 Spark image materially improve the fixed
single-request, three-machine path?

Only the Spark image changed:

- previous: `sha256:446a23697c7eba9e0434cb162b5389792c7a5ab0f2652173bf1e791e7fdb4c5d`,
  patch label `5e01f1bb74d8`;
- candidate: `sha256:e97db95dd4b3f9a5ebd6ddeafb8d7d422dda729cc01b33d7f5c6ecfcf9cf2aaf`,
  patch label `9f73cca659a1`, repository digest
  `sha256:ef83797d791fef96c4605e8d37367aca6de5aeac7bb672792cb682e2e55d4237`.

The portable model, Mac runtime, protocol 4, 96 MiB frames, C1 prompt, greedy
400-token output, four-bit DFlash depth two, and five cold runs stayed fixed.

## Result

Status: **retained for currentness; failed the performance goal**.

| Measure | Current Mia image | Prior image |
| --- | ---: | ---: |
| Median complete | 6.907425 s | 6.912239 s |
| Median TTFT | 0.396782 s | 0.397171 s |
| Median decode | 61.270 tok/s | 61.118 tok/s |
| Median handoff | 0.385920 s | 0.390569 s |
| Maximum complete | 9.632238 s | 6.945361 s |
| First differing token | 104 | 104 |

The roughly 5 ms median change is not a material single-stream improvement.
The candidate remains `0.334744` s above the required `6.572681` s ceiling.
The first request's `3.112706` s TTFT exposes the still-open first-use
compilation gate.

MCDMA calls advanced from 0 to 25 on each side with zero failures. Both Spark
containers used the exact candidate image and `tf.patches=9f73cca659a1`, and
both reported `OOMKilled=false`. Cleanup acknowledged shutdown and left no
owned Spark, Studio, or MCDMA process running.

## Evidence

Raw directory:

`results/raw/mia-c1-three-machine-cold-dflash-progressive-96m-20261004T083157Z-70321`

- `result.json`: `1a4d4e61eea98bd28d58a71dca86ab8ac7354584469d167623ae3ef9928df5df`
- `summary.json`: `248968f4fd59ab22fa78eabf2c0e77305e9c8cc922c02cc00487a63ba2bb4b70`
- `head.log`: `bbc48700ba69a7e94e9396e60a2bee503c9b57d6f4e18d876e91c7ba0b8b19ce`
- `worker.log`: `854efc9e53cfa7f1bbd31eebd7e531993a3cef3fa21a2f3143771a4e70d24af6`
- shutdown evidence:
  `results/raw/stopped-cache-handoff-20261004T083336Z-70852`

## Decision

Keep the current Mia v1.5 image because it is compatible and current. Do not
claim its concurrency and cancellation changes as a one-stream speedup. The
next experiment must address the token-104 numerical split or Mac target
forward cost rather than repeat image or MCDMA direction tuning.
