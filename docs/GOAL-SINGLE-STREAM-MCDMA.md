# Goal: one faster request across two Sparks and one Mac Studio

Date: 2026-10-03

## Objective

Make one GLM-5.3-Flash request faster than MiaAI-Lab's published two-Spark
TensorFold one-request claims:

```text
client request
  -> both Sparks prefill the prompt
  -> MCDMA moves the complete live prompt cache
  -> Mac Studio decodes the reply
```

The candidate uses Mia's exact sparkDash C1 prompts, greedy settings, thinking
mode, warmup, and 400-token output budget. It passes only when its complete
client-observed time is at least 3% below the time implied by Mia's published
TTFT and decode rate. Exact token IDs are still checked against the retained
same-checkpoint control, but that control is no longer the performance target.

The sixteen-request result in `GOAL-CONCURRENT-THREE-MACHINE.md` is a valid
capacity result, but it is not acceptance for this goal. Running eight requests
on the Sparks and eight on the Mac proves combined capacity, not a faster single
request.

## Current result

This goal is **not yet achieved**.

- The current Mia v1.5 Spark image, ID `e97db95...` with patch label
  `9f73cca659a1`, passed the real three-machine path. Five cold requests reached
  `6.907425` s median complete time, `0.396782` s median TTFT, `61.270` decode
  tok/s, and `0.385920` s median handoff. The first request took `9.632238` s
  because of first-use compilation. Output still first differed at token 104;
  MCDMA recorded zero failures and neither Spark was OOM-killed.
- That current image is retained for upstream currency, not as an optimization:
  its steady median was only about 5 ms faster than the earlier image. The goal
  still misses the required ceiling by `0.334744` s. Protocol 4 and the 96 MiB
  two-frame Mac-pull shape remain retained.
- A controlled Spark-push versus Mac-pull comparison preserved the same cache,
  prompt, output hash, frame shape, and zero-failure transport. Push reduced the
  median handoff by about 3 ms but completed in `6.917320` s, 5 ms slower than
  pull. Reply direction is therefore closed as an end-to-end tuning axis and
  Mac-pull remains retained.
- Both campaigns shut down both Spark ranks, the Studio connector, and the
  MCDMA link cleanly with zero MCDMA failures and no OOM indication.

- The first fair single-request campaign used Mia's C1 prose prompt, warmup,
  greedy settings, and 400-token budget. It had zero MCDMA failures and a
  6.772827 s median complete time, essentially level with Mia's implied
  6.775960 s claim but slower than the required 6.572681 s ceiling.
- That campaign failed correctness: its answer first differed from the local
  same-checkpoint control at token 52.
- A Mac-only diagnostic proved the original handoff boundary itself was wrong:
  it changed the answer at token 24 without CUDA, networking, or MCDMA.
- A second Mac-only diagnostic cached the complete prompt and then continued
  from the first reply token. It matched the ordinary local answer exactly.
- Handoff protocol v2 now implements that corrected boundary. The Sparks cache
  the complete prompt and export their first-token reference plus the final
  hidden row; the Mac computes and validates that first token before decoding
  the remainder. Unit and lifecycle tests pass and the live campaign has been
  rerun.
- The best complete cold-path MTP result was 7.054299 s median, 0.541085 s
  median TTFT, and 61.367 decode tok/s. The positively attested DFlash2 rerun
  reached 7.048070 s median, 0.542225 s TTFT, and 61.316 decode tok/s. Its
  startup log says `DFlash2 block`, every run contains nonzero DFlash telemetry,
  MCDMA had zero failures, and both Sparks remained free of OOMs. It still
  missed the required 6.572681 s ceiling by 0.475389 s.
- Removing redundant reply, payload, and import copies reduced steady-state
  Mac import to 0.0049-0.0085 s. The steady handoff itself remains about
  0.528-0.543 s, so further full-buffer copy tuning cannot close the gap.
- A warm-prefix diagnostic reached 6.516108 s, but it excluded prompt setup
  and transfer and therefore is not the accepted end-to-end result.
- A one-load Mac sweep tested four-bit DFlash2 depths 1 through 7. Depth two
  was fastest at 6.914866 s complete and 60.009 decode tok/s. Eight-bit depth
  two was slower at 6.998244 s and 59.285 decode tok/s. More accepted guesses
  did not offset their added work.
- With measured 0.542225 s TTFT, the Mac would need 66.164 decode tok/s to hit
  the 3% complete-time ceiling. Current 61.316 is 7.91% short. Even an
  idealized path retaining only measured 0.185 s Spark prefill and 0.096 s
  wire call needs 63.417 tok/s. The compute-ceiling stop applies: this short
  C1 request cannot pass through another cache-copy optimization alone.
- The current answer still first differs from the Mac-only same-checkpoint
  control at token 104. The official two-Spark same-checkpoint diagnostic also
  differs deterministically, so this is recorded as a cross-backend numerical
  mismatch rather than an MCDMA corruption claim.
- A 19-variant cache transplant diagnostic localized that mismatch. Replacing
  all 45 prompt-layer caches with Mac-local state matched all 399 checked
  next-token decisions; the draft cache was irrelevant. KDA-only, MLA-only,
  and every early/middle/late family partition still diverged. A small Mac
  recompute cannot repair exactness: doing so requires numerical parity in the
  CUDA and Metal prompt kernels or repeating the complete Mac prefill, which
  would erase the intended prefill advantage.
- The older four-request handoff remains useful historical evidence: its Mac
  decode was 92.87 aggregate tokens/s, but the complete path was 52.64 versus
  121.63 aggregate tokens/s for the matching two-Spark run.
- The current handoff omits sparse-attention index state and is limited to
  roughly 2,051 total tokens. A native two-Spark service supports much longer
  contexts.

The published Mia number is the external performance target; no separate Mia
checkpoint download is required. The retained portable checkpoint is used only
to prove that the candidate preserves its own greedy output.

The pinned research and exact control definitions are in
`RESEARCH-MIA-TONY-SINGLE-STREAM.md`.

## Scorecard

| Measure | What it tells us | Pass rule |
| --- | --- | --- |
| End-to-end latency | What the person actually waits for, from request start to final token | Candidate median at least 3% lower, with the paired lower confidence bound also above 3% |
| Time to first token | How long the blank wait lasts | Report it from client request start; no hidden regression greater than 2% |
| Spark prefill time | Time both Sparks spend reading the prompt | Same prompt and measurement boundary in both arms |
| Cache export time | Cost of turning live Spark state into transferable state | Measured separately and included in candidate latency |
| MCDMA transfer time | Time spent moving the cache | Measured bytes, seconds, and effective Gbit/s; zero failures |
| Mac import time | Time spent making the transferred cache usable by Metal | Every copy counted; included in candidate latency |
| Decode speed | Reply-generation speed after the first token | Tokens/s plus token-to-token p50, p95, and p99 latency |
| Context length | How much prior text the service can really use | Test 2K, 8K, 32K, 64K, and 128K; report the actual admitted maximum |
| Cache footprint | Why a context length fits or fails | Cache format, bytes per token, total bytes, and precision |
| Output equality | Whether the split changed the answer | Exact token-ID and output hash match against the same-checkpoint control; fixed quality suite against Mia's EXL3 control |
| Published recipe | Whether the claimed win beats the requested real reference | Compare against MiaAI-Lab v1.5's published C1 TTFT and decode rate using the exact sparkDash prompts and request settings |
| Spark clocks | Whether one degraded rank silently slows the pair | Both ranks recorded under load; a materially degraded rank invalidates the round |
| Memory safety | Whether speed comes from paging or instability | Peak Mac wired/active memory, Spark available memory, zero swap pressure, OOM, or restart |
| Recovery and cleanup | Whether a failure leaves the machines usable | Bounded recovery and no owned process, container, socket, or helper left running |

The primary result is end-to-end latency, not tokens/s by itself. Decode
tokens/s diagnoses the compute ceiling. Time to first token diagnoses the
prefill and handoff tax. Context and cache measurements prevent a short-context
demo from being presented as the full service.

## Break-even rule

Let:

- `P` be the two-Spark prompt-prefill time;
- `S` be the two-Spark time per decoded token;
- `M` be the Mac time per decoded token;
- `H` be cache export, MCDMA transfer, and Mac import time; and
- `N` be the number of output tokens.

The baseline is `P + N*S`. The candidate is `P + H + N*M`. For a 3% win:

```text
H + N*M <= 0.97 * (P + N*S)
```

Because both arms already use the Sparks for prefill, prefill does not create
the speedup. The Mac must decode enough faster to pay the complete handoff cost
and the 3% margin. If `M >= S`, this design cannot win without overlapping or
removing work that the baseline also performs.

## Frozen comparison

The same-checkpoint control and candidate use:

- one request at a time;
- the same content-addressed model snapshot on all three machines;
- the same tokenizer and rendered prompt token IDs;
- greedy generation, thinking off, and the fixed output budget for that
  campaign;
- identical cold or warm state within each paired round;
- counterbalanced `A B B A B A A B` ordering;
- one warmup followed by measured requests; and
- fresh process/container identities where the campaign calls for a restart.

The Mia comparison cells use sparkDash's 32-token warmup and 400-token C1 prose
and structured replies. Mia's claims imply 6.77596 seconds total for prose and
3.62764 seconds for structured; the 3% pass ceilings are respectively 6.57268
and 3.51881 seconds. The initial scientific context sweep is 2K,
8K, 32K, 64K, and 128K prompt tokens. A
candidate that supports only 2K is reported as a short-context prototype and
cannot close the project.

## Work order and stop conditions

1. Pin current compatible TensorFold, Mia recipe, sparkDash, tonyd2wild
   research, MCDMA, and model identities.
2. Freeze MiaAI-Lab v1.5's published C1 prose and structured targets and exact
   sparkDash request method; do not download its checkpoint solely to restate
   those claims.
3. Measure the same-checkpoint correctness control and a Mac-only decode
   ceiling on the exact portable checkpoint.
4. Stop transport optimization if the Mac cannot decode enough faster than the
   Sparks to satisfy the measured break-even equation.
5. If the compute ceiling is viable, remove avoidable cache copies by using a
   persistent registered MCDMA buffer and direct Metal/MLX import.
6. Export and import the real sparse-attention index state and MTP state. Exact
   tokens must pass before contexts above 2K are timed.
7. Run the frozen repeated campaign against the published claim thresholds.
   Keep every failed attempt and change one
   setting at a time.
8. Accept only after the speed, output, context, memory, recovery, and cleanup
   gates all pass twice from clean starts.

Mac-side DFlash2 is not treated as a cosmetic setting. The reviewed Mac GLM
implementation is now part of the isolated TensorFold stage, but a benchmark
counts as DFlash2 only when startup reports `DFlash2 block` and measured rounds
contain nonzero DFlash telemetry. A requested command-line path alone is not
runtime proof.
