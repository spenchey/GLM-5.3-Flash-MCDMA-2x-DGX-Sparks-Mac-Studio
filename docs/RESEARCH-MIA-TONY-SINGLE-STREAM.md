# Research basis for the single-request three-machine recipe

Date: 2026-10-03

## Correct comparison

This project is not an eight-request capacity exercise. Its acceptance case is
one request that uses both DGX Sparks for prompt processing, MCDMA for the live
cache handoff, and the Mac Studio for reply generation.

The candidate has two evidence references:

1. **Published target:** MiaAI-Lab release v1.5's claimed one-request TTFT and
   decode rate, using its exact sparkDash prompts and request settings.
2. **Same-checkpoint correctness control:** the Mac-compatible checkpoint
   served entirely by the two Sparks, used to verify exact token output.

The three-machine candidate must beat the complete time implied by Mia's claim
by at least 3%. Token and text equality is required against the same-checkpoint
control. Mia's published receipt has no output bytes to compare, so the external
comparison uses the same prompt, generation settings, output budget, and a
fixed quality suite rather than pretending an unavailable hash exists.

## MiaAI-Lab control, pinned

Source: [MiaAI-Lab GLM-5.3 TensorFold recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)

- measured v1.5 recipe commit: `1576746a04983b6eded0551dbf22512ee9e95654`
- current main checked on 2026-10-03: `a15282478bd0e58ea80c512c330df3f61af85f74`
  (the later commit is documentation-only; the image, pins, defaults, and C1
  numbers below are unchanged)
- recipe release: v1.5, dated 2026-10-03
- TensorFold base: v0.6.0 plus 70 recipe patches
- image: `v0.6.0-9f73cca659a1`
- image digest: `sha256:ef83797d791fef96c4605e8d37367aca6de5aeac7bb672792cb682e2e55d4237`
- checkpoint: `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`
- checkpoint revision: `078455ffe6472f9a52fbc1139f58b9db2881b25c`
- drafter: `incoai/GLM-5.3-Flash-DFlash2`
- drafter revision: `bf582e4eacc1810f76656d1811693ff6c6737d2a`
- defaults: four request slots, FP8 cache, q4 dense weights, DFlash2, full
  1,048,576-token request window, RoCE between the Sparks

Mia reports one-request results of 60.4 tok/s and 170 ms TTFT for its prose
prompt, and 114.7 tok/s and 149 ms TTFT for its structured count prompt. For
399 post-first-token intervals, those claims imply 6.77596 and 3.62764 seconds
of complete request time. A 3% win therefore requires at most 6.57268 and
3.51881 seconds. The owner chose these published values as the external target;
another roughly 176 GB model download is not required.

Mia v1.5's eight-request change explicitly says one-request speed is unchanged.
It therefore does not alter this project's target or justify using aggregate
eight-request output as acceptance.

## Mia's measurement method, pinned

Source: [MiaAI-Lab sparkDash](https://github.com/MiaAI-Lab/sparkDash), commit
`b4228a330a7877dcb5a30516500d57e26affa45a`.

For each one-request prompt type:

- one 32-token warmup is discarded;
- the measured reply requests 400 tokens;
- temperature is 0, top-p is 1, and thinking is explicitly disabled;
- the request is streamed from a different machine;
- TTFT is request start to first visible token;
- decode speed is `(completion tokens - 1) / (last token - first token)`;
- complete request time is request start through the finished response.

The two required C1 prompts are:

- prose: the fixed detailed hash-map explanation prompt;
- structured: count from 1 to 200, numbers only.

Structured output is unusually draft-friendly. It is kept as a compatibility
check, but prose plus code and real agent-style prompts prevent a misleading
win on one easy output shape.

## tonyd2wild lessons

Source: [tonyd2wild GLM-5.3 DGX Spark Cookbook](https://github.com/tonyd2wild/GLM-5.3-DGX-Spark-Cookbook),
commit `f72b0ddfd491c815027f9b56c82af4866f24e01b`, and the current
[two-Spark EXL3 recipe](https://github.com/tonyd2wild/GLM-5.3-Flash-EXL3-on-2x-NVIDIA-DGX-Spark),
commit `dc91a125fc60349ce99498d65dac5bc772a43c54`.

The reusable practices are:

- report wall-clock request time, decode speed, TTFT, context limit, cache
  format, cache capacity, and memory headroom together;
- pin image, model, cache size, and settings so boots are comparable;
- check every Spark's GPU clock under load because one degraded rank slows the
  whole pair;
- do not trust a successful boot as proof against unified-memory pressure;
- preserve failed experiments and measure one change at a time;
- lead with real prose/code prompts and label counting prompts only as an easy
  speculative-decoding ceiling;
- run the two lanes isolated, from the same client and time window, and keep
  the raw receipts rather than comparing numbers produced by different tools.

Tony's vLLM numbers are not the control for this project. His work supplies the
measurement and failure-detection discipline; MiaAI's TensorFold recipe is the
control requested by the owner.

The current two-Spark GLM repository was also checked at commit
`dc91a125fc60349ce99498d65dac5bc772a43c54`. Its reusable additions are to
warm the relevant request shapes, count completion tokens from usage rather
than stream chunks, preserve failed trials, and pin cache sizes only after
measuring the actual cache pool and memory headroom. Its performance figures
remain research context, not this project's comparator.

## MCDMA and related handoff evidence

Source: [MCDMA](https://github.com/ashhart/MCDMA), commit
`e672c14ff9fc7b38994caf73025cf1588b4de74e`.

MCDMA has measured host-memory transfers and one first-version Qwen cache
handoff. Its own documentation warns that direct registration of existing CUDA
allocations and direct access to Metal private buffers are not proven. The
verified direction is preallocated compatible shared buffers. Therefore this
project counts cache export, every copy, transfer, import, and first Mac decode
separately; negotiated 100 Gb/s is never reported as application throughput.

Independent Spark-to-Mac prefill/decode work also found that importing the
cache with the wrong dtype can put MLX on a slow path. Our candidate must prove
the Mac cache precision and compare decode speed before and after injection.

## Required result table

Every accepted prompt/context cell records these three rows:

| Row | Machines and work |
| --- | --- |
| Mia v1.5 published target | claimed TTFT, decode rate, and their implied complete time |
| Same-checkpoint correctness control | both Sparks perform prompt and reply using the candidate's exact checkpoint |
| Three-machine candidate | both Sparks process the prompt, MCDMA moves the complete cache, Mac Studio generates the reply |

The primary number is complete client-observed request time. Decode tok/s,
TTFT, prompt processing, transfer, cache size, token gaps, context, memory,
clocks, and error counters explain the result but cannot replace it.

Mac-only and same-checkpoint two-Spark measurements are diagnostic controls,
not acceptance arms. They may locate a bottleneck or prove output identity,
but the only candidate eligible to beat Mia is the live path that uses both
Sparks for prefill, MCDMA for the cache transfer, and the Mac Studio for decode.
