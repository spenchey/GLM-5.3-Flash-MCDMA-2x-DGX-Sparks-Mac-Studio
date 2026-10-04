# Expert source map

This file ranks the evidence used to improve the two-Spark plus Mac Studio path.
Measured evidence from the exact three machines outranks a generic tuning claim.

## 1. This project's measurements

- `docs/EXPERIMENT-LOG.md` records every accepted and rejected change.
- `results/receipts/` holds compact evidence that can be reviewed later.
- `results/raw/` holds ignored machine output with recorded SHA-256 hashes.
- `docs/GOAL-SINGLE-STREAM-MCDMA.md` freezes the comparison and pass criteria.

These records identify the current bottleneck. With the current Mia v1.5 Spark
image, protocol 4 reached a 0.386-second median handoff and a 6.907-second
median complete request, but the accepted ceiling is 6.573 seconds. The image
change itself was neutral: about 5 ms faster than the prior measured image.
The Mac target-model decode remains about 61.27 tokens/s, the reply still first
differs at token 104, and the current handoff is only a short-context prototype.
MCDMA reply-direction tuning did not improve complete time.

The cache-transplant diagnostic rules out a cheap exactness patch. Only using
Mac-local state for all 45 prompt layers matched the local path for all 399
checked decisions. Neither cache family alone worked, and leaving any tested
early, middle, or late KDA or MLA group imported eventually diverged. Do not
repeat partial cache transplantation; exactness work now belongs at the
CUDA-versus-Metal prompt-kernel boundary.

That boundary is now narrower. A layer-0 diagnostic made its input and all
three projections byte-exact across CUDA and Metal. Disabling CUDA's faster
chunked KDA did not repair the recurrent state: its relative L2 remained
`0.00558999`, and the first wrong teacher-forced token merely moved from 81 to
90. Combining those exact projections with the first Mac storage-boundary
patch improved state error about 3.4% to `0.00539980`, but token 90 remained
wrong. The captured bfloat16 value immediately after convolution still differed
at relative L2 `0.00393108`. Source comparison explains why: Metal uses its
stable precise sigmoid instantiated on bfloat16, while the first CUDA patch
used a float `x / (1 + exp(-x))` expression with only outer casts. The next
bounded test is that activation and beta arithmetic alone; broad transport,
cache-transplant, projection, storage-cast, and serial-KDA sweeps are closed.

## 2. TensorFold source and history

- Source: <https://github.com/ashhart/TensorFold>
- Official release pin: TensorFold `0.6.5`, commit
  `609ca419abecebdc5a059498a613680bd3aa847f`, recorded in `UPSTREAM.lock`.
- Local inspection checkout: sibling `../TensorFold`. It is currently on the
  isolated experiment branch `codex/glm-mac-dflash2` at `8debc4e`, one local
  commit above the official pin. That extra commit is this project's Mac
  DFlash2 experiment, not an upstream TensorFold release or owner claim.
- Highest-value areas: GLM MLX kernels, family round allocation, DFlash
  integration, exact-output tests, release commits, and rejected experiments.

Start with these concrete files rather than reading the repository front to
back:

- `src/tensorfold/families/glm5_next/runtime.py`: the Mac target forward,
  imported-cache path, MTP, and DFlash behavior;
- `src/tensorfold/families/glm5_next/cuda/drafter_choice.py`: the existing
  measured CUDA selector that keeps both MTP and DFlash available;
- `src/tensorfold/families/glm5_next/cuda/decode.py`: exact verification,
  cache ownership, and committed-token accounting;
- `src/tensorfold/families/glm5_next/cuda/forward.py`: prompt/decode buffers and
  DFlash tap production;
- `git log -- src/tensorfold/families/glm5_next`: the owner's sequence of
  measured changes, including ideas that were deliberately rejected.

TensorFold's own measured row costs and exact-output checks are the authority
for changes to Mac decoding. Release claims alone are not performance proof.
The complementarity experiment found 11 rounds where the checkpoint's MTP head
predicted the first token and DFlash did not, and 10 rounds with the opposite
result. Running both heads every round reduced decode to about 57.94 tok/s. A
second experiment ported TensorFold CUDA's measured one-head-per-round selector
to the Mac; it preserved the exact output but reached only 57.88 tok/s versus
the retained DFlash-only 60.54 tok/s. Historical payoff cannot predict which
head will win the next round, and periodic MTP probes cost more than they save.
Do not repeat selector tuning without a genuinely predictive signal.

### Closest independent TensorFold optimization lab

- Source: <https://github.com/jayleaton/glm53-tensorfold-spark>
- Start with `docs/PROFILE.md`, `docs/RESULTS.md`, `results/theory2/`, and the
  test windows recorded beside each retained or rejected patch.

This is the most useful outside reference for kernel-level GLM work. Its
captured two-Spark profile attributes about 44% of a one-stream decode round to
routed experts, measures exposed communication and overlap tax separately, and
records profiler distortion. Mia's measured L2-prefetch and non-coherent EXL3
load patches were adapted from this work. Use its measurement method and
kernel attribution; do not copy CUDA settings onto Metal without a Mac trace.

## 3. DFlash paper and implementation

- Paper: <https://arxiv.org/abs/2602.06036>
- Code: <https://github.com/z-lab/dflash>

DFlash explains parallel draft generation, target verification, and why draft
acceptance must be weighed against a wider target pass. Its reported multi-fold
speedups are model- and workload-specific; they are not a GLM guarantee.

## 4. MCDMA source and lab reports

- Source: <https://github.com/ashhart/MCDMA>
- Installation and measured hardware: <https://github.com/ashhart/MCDMA/blob/main/docs/install.md>
- Closest architectural precedent:
  <https://github.com/ashhart/MCDMA/blob/main/docs/disaggregated-inference.md>
- KV handoff ownership and wire contract:
  <https://github.com/ashhart/MCDMA/blob/main/docs/kv-handoff.md>
- Official project pin: commit
  `e672c14ff9fc7b38994caf73025cf1588b4de74e`, recorded in `UPSTREAM.lock`.
- Local inspection checkout: sibling `../MCDMA`. It is currently on the
  isolated experiment branch `codex/v3-payload-relay` at `ae96ca3`, one local
  commit above the official pin. That extra commit is this project's payload
  relay repair, not an upstream MCDMA owner change.

The published MCDMA latency work covers one directly connected Spark and one
Studio. This project's second Spark and TensorFold cache format therefore need
their own end-to-end proof. Nominal 100 Gb/s link speed is not application time.
The Qwen disaggregated-inference note is especially useful because it measures
the same job split: Spark prefill, MCDMA cache pull, and Mac decode. It reports
the exact stage costs, cross-backend cache similarity, break-even behavior by
prompt length, and the remaining persistent-buffer, direct-import, and
progressive-transfer work. Its vLLM connector is a protocol reference, not a
drop-in TensorFold implementation.

## 5. GLM architecture and official serving references

- Model and architecture: <https://huggingface.co/zai-org/GLM-5.3-Flash>
- Official vLLM recipe: <https://github.com/vllm-project/recipes/blob/main/models/zai-org/GLM-5.3-Flash.yaml>
- Official SGLang cookbook: <https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx>

These sources establish model shape, attention behavior, generation settings,
and comparison practices. They do not validate this TensorFold/MCDMA path.

## 6. Published community recipes

MiaAI-Lab and tonyd2wild are useful for reproducible launch files, prompt
fixtures, version pins, and reporting format. Their published two-Spark result
is the frozen external target; their checkpoint is not required locally merely
to compare against that claim.

Mia's current `v1.5` history is also a concise index of measured changes. Read
the patch table rather than only its headline numbers: prompt kernels and
overlap, decode kernels, DFlash2, RoCE small-message latency, L2 prefetch,
non-coherent EXL3 loads, and kept-prefix behavior each name their measured
effect. Concurrent-stream gains do not establish a one-request gain.

Currentness was rechecked on 2026-10-04. MiaAI-Lab `main` was
`c01562baf805361e9b8911405a6010b5cdd6100d`, tonyd2wild `main` was
`f72b0ddfd491c815027f9b56c82af4866f24e01b`, and jayleaton `main` was
`eae051adef8790cfcde56a3772a16df52f1ed45e`. Mia's changes after the pinned
v1.5 baseline add TP3 address/qualification work, not a new two-Spark
single-request kernel result. Recheck these heads before copying a future
change.

## 7. Disaggregated-prefill designs

- vLLM design: <https://github.com/vllm-project/vllm/blob/main/docs/features/disagg_prefill.md>
- KV transfer interface: <https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/README.md>

These references show how established engines separate prompt work from reply
work and define producer/consumer ownership. They are design references, not
evidence that this mixed CUDA/MLX path is faster.

## 8. Apple GPU profiling

- MLX Metal capture: <https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html>

This is the primary profiling route for the remaining Mac target-forward work.
The current Studio has the MLX capture API, but its active developer tools are
the command-line package rather than full Xcode, so a captured trace cannot yet
be inspected with Apple's GPU tools on that machine.

## 9. Inference-engineering method

- Book: Philip Kiely's *Inference Engineering*:
  <https://www.baseten.co/inference-engineering/llms.txt>
- Project extraction: `docs/RESEARCH.md`
- Frozen one-variable evaluator: `docs/AUTORESEARCH.md`

Use this for the method rather than for model-specific numbers: freeze the
baseline, measure TTFT, decode rate, end-to-end time and tails, change one
thing, then keep or reject it. It also explains why speculative multi-token
decoding is a larger possible lever than shaving a small transport cost.

## 10. Adversarial reviews and failed ideas

- Performance review:
  `results/receipts/2026-10-02-fable-performance-review.md`
- Architecture review:
  `results/receipts/2026-10-01-fable-adversarial-review.md`
- Append-only results: `docs/EXPERIMENT-LOG.md`

These are useful because they record why attractive ideas were rejected. They
also prevent a fast but wrong answer, an unproven machine path, or a warm-cache
shortcut from being called an optimization.

## Practical limit

A ten-fold improvement to one request on the same model and hardware cannot be
promised by a setting change. It would require a different model, a major new
decode algorithm, or a different objective such as serving many requests at
once. The immediate objective remains the measured one: make the three-machine
single request at least 3% faster than the published two-Spark target without
changing its output.

The fastest route to expertise is therefore: read this project's measured
failures first, trace the exact TensorFold hot path second, use the DFlash and
disaggregated-prefill designs to propose one change, then require a repeated
exact-output measurement on the real three machines. Reading more sources
without that loop will not improve the service.

The next discriminating experiment is not another broad tuning sweep. Keep the
now-exact first-layer projections and reproduce only the Mac prompt kernel's
stable bfloat16 sigmoid arithmetic for convolution activation and beta on CUDA.
Require those captured stages to become exact before changing normalization,
decay, or the state update. After exactness is settled, profile the Mac target
forward in Metal; that identifies the decode work that must improve to close
the measured speed gap.
