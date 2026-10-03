# Three-machine success and optimization plan

Date: 2026-10-02

> The single-request layer-split campaign below is retained as the original
> correctness plan. `GOAL-CONCURRENT-THREE-MACHINE.md` now owns the active
> capacity goal and its same-workload two-Spark comparison.

## Objective

Deliver one persistent GLM-5.3-Flash service in which the Mac Studio runs the
tokenizer and original model layer 0, MCDMA carries every activation and reply,
and both DGX Sparks run original layers 1 through 44 together. The old
two-Spark service is not the target and cannot close this project.

The order is fixed: correctness, proof of all three machines, recovery,
measurement, then optimization. A faster wrong or fragile service is a failed
candidate.

## What success means

| Gate | Pass condition |
| --- | --- |
| Correctness | Repeated identical requests return identical token IDs; both Spark ranks select the same token at every step; fixed canaries and a reference prompt set match the pinned TensorFold 0.6.0 engine. |
| Answer quality | The split causes no model degradation. Exact canaries pass, the reference tokens match, and the bounded end-of-turn probe does not exhaust its token allowance. |
| Three-machine participation | For every request, Mac Metal time is nonzero, both Spark containers are running before and after, and the MCDMA call count rises by `2 + 2 * (prompt chunks + generated tokens - 1)`. |
| Stability | Zero MCDMA failures, rank disagreements, protocol errors, stalls, or growing memory during the load and idle soaks. |
| Recovery | Each written fault is recovered by the owned script inside the measured budget, followed by an exact canary and zero orphaned processes, sockets, helpers, or locks. |
| Reproducibility | Every result records the project commit, deployment ID, image ID, model/stage hashes, configuration hash, performance knobs, host state, and raw sample hashes. |
| Operational cleanliness | Normal stop does not require `SIGKILL`; owned temporary state and helpers are removed; evidence remains preserved; the final service is healthy. |

## What good performance means

This is a one-user, non-streaming service. Latency and consistency matter more
than aggregate throughput.

Primary measures, in order:

1. steady decode-step median and worst-step latency near positions 100 and 1000;
2. time to first token for 64-, 512-, and 2048-token prompts;
3. end-to-end time for fixed short, long-answer, and long-context requests;
4. decode tail latency: p99 no more than 1.3 times the median, no more than two
   hiccups above twice the median in a round, and zero stalls above the larger
   of four times the median or 150 ms;
5. zero errors and flat process/host memory across the soak;
6. measured recovery time after each deliberate fault.

The current approximately 31-token/s result is only a provisional observation
from the proof-first, one-token-per-round implementation. It is not the final
target. The unchanged local two-Spark MTP service measured 66.2 output tokens/s,
so the three-machine service is currently about 53% slower. MTP is part of the
real comparison, not an excuse to exclude it: the final three-machine service
must preserve multi-token prediction and beat a fresh same-version two-Spark
baseline. The historical result implies a minimum provisional target of 68.2
tokens/s, but the binding target is the fresh baseline plus the paired 3% lower
confidence bound below. The five-token canary rate is not a speed measure because
setup cost dominates it.

Existing measurements show approximately 7% of a decode step in Mac layer 0,
92% in the Spark-side compute round trip, and 1% in acknowledgement. Therefore
the first optimization work must measure the Spark phase instead of assuming
the network is the bottleneck.

For this service, "good" does not mean a vanity token/s number. It means B2's
final-128 median is no more than 1.2 times B1, all tail limits above pass, every
TTFT shape repeats within 2%, the first request after an idle interval is no
more than twice steady state and the second is back within 2%, memory stays
within 64 MiB or 1% after warmup, and a candidate clears the paired improvement
rule below without any correctness, safety, or secondary-metric regression.

## Fresh upstream evidence

| Source | Reviewed identity | Relevant owner change | Decision |
| --- | --- | --- | --- |
| TensorFold | `v0.6.3`, `9356df5c424b0c36b7737e37873a6f968b08de79` | MLX shared rounds admit up to eight requests; owner benchmark checks TTFT, aggregate rate, and concurrent-vs-solo hashes | Use on the Mac in an isolated, hash-pinned stage; do not substitute it into Mia's CUDA image. |
| Mia GLM recipe | `cf28cc4f8038be322cdeda220c6f1c8ace8f27d1`, image `v0.6.0-5e01f1bb74d8` | 68-patch current GLM CUDA image, concurrent streaming fixes, twin rails, and optional three-Spark work | Use its exact published image on both Sparks; measure against a fresh same-image baseline. |
| MCDMA | `e672c14ff9fc7b38994caf73025cf1588b4de74e` | Stock-Linux endpoints and documentation; no post-pin `rpc/` change | Keep the hardware-proven RPC binaries. Treat keepalive only as a later idle-wake experiment. |

## Correctness work before measurement

1. Make the Linux/listen mailbox follow the real MCDMA contract. Its local
   ready and generation words are zero. Bind the first authenticated frame's
   positive generation, reject later changes, and use the daemon control socket
   as Linux-side link-liveness proof.
2. Seed each new Mac client sequence with unpredictable entropy mixed with the
   visible mailbox words, and burn every published sequence.
3. Keep the authenticated full-frame SHA-256 as the payload-integrity gate.
   A control-word reread detects a concurrent rewrite but cannot prove that
   payload bytes were never overwritten.
4. Enforce the CUDA engine's actual dense-context limit of 2051 positions on
   Mac and both Sparks before any collective work. Do not advertise 4096 until
   long-context sparse state is actually implemented.
5. Fix recovery's stale PID handling, command predicates, launch return-code
   handling, and deploy/preflight quoting. Add real constructor and failed-state
   tests instead of mocks that bypass the broken path.

## Measurement prerequisites

Before tuning, the deployed service and benchmark must emit and preserve:

- Mac timing arrays for tokenize, mailbox open/reset, Metal, Spark/MCDMA round
  trip, acknowledgement, first token, and every decode step;
- Spark rank-0 timing for frame validation, control collective, activation copy
  and collective, model layers, head/token agreement, and reply;
- health identity: deployment/config/image IDs, performance knobs, completed
  and failed request counts;
- benchmark identity, host clocks, temperature/power where available, memory,
  MCDMA call/failure deltas, and both container states before and after;
- committed B1-B4 benchmark prompts and E1-E8 equivalence prompts with hashes;
- a one-time greedy, MTP-off reference-token capture from the pinned native
  TensorFold 0.6.0 image during a maintenance window.
- client request-receipt-to-first-token timing, Mac RSS and MLX active/peak
  memory, Spark host MemAvailable and process RSS, both container environments,
  rank-1 committed-window evidence, and head/worker logs joined to each request;
- a one-request profiler trace that separates in-layer NCCL work from CUDA
  layer/head compute, plus a private-link 8 KiB all-reduce microbenchmark.

## Baseline and comparison rule

Establish three restart-separated B1 baseline rounds. Each round has one warmup
and five measured requests. The warmup is not a sample. The three round medians
must differ by at most 2%; otherwise stabilize the hosts before tuning.

| Test | Shape | Primary measure |
| --- | --- | --- |
| B1 decode | about 20 prompt tokens, 128 output tokens | steady step median and maximum |
| B2 long decode | about 20 prompt tokens, 1024 output tokens | final-128 step median and maximum |
| B3 first token | 64, 512, and 2048 prompt tokens, 1 output token | end-to-end first-token time |
| B4 realistic | about 300 prompt tokens, 256 output tokens | end-to-end time |
| E1-E8 equivalence | code, arithmetic, extraction, French, lists, and long answer | exact token sequence against the pinned reference |

- Screen exactly one changed setting with eight restart-separated rounds in
  `A B B A B A A B` order. Every round has one warmup and five measured B1
  requests. Pair adjacent rounds and use each round's decode-step median as the
  sample; individual requests are not independent samples.
- Compute the paired log-ratio. All four pairs must favor B, and the mean gain
  minus `2.353 * standard_error` must be at least 3%. If mean gain is above 3%
  but that bound fails, extend once with four counterbalanced rounds and decide
  at six pairs using multiplier `2.015`. Never extend twice.
- Confirm a screened candidate with four restart-separated rounds in
  `B A A B` order. Every confirmation round runs B1, B2, B3 at 64, 512 and no
  more than 2040 total tokens, B4, and E1-E8. No B2/B3/B4 paired mean may
  regress more than 2%.
- All reference hashes, identities, call deltas, rank counters, health gates,
  and memory limits must pass before performance is considered. Re-establish A
  after every accepted change. A rejected candidate's restored round must be
  within 2% of the neighboring A rounds.

## Optimization order

1. Add the missing instrumentation, frozen token oracle, B1-B4 prompts and a
   fresh same-version two-Spark MTP baseline. Run timing on and off once to
   quantify its observer effect; normal baselines keep synchronized phase timing
   off.
2. Preserve TensorFold's MTP drafter and change the three-machine protocol from
   one token per Mac/Spark round trip to one verification window per round trip.
   The Sparks draft future tokens, the Mac runs layer 0 over the whole candidate
   window, both Sparks verify it, and all three machines partially commit the
   same accepted prefix. Exact token identity and cache position must pass before
   any speed result is considered.
3. Measure draft cost, verification-window time, accepted tokens per round, and
   acceptance rate. At the current approximately 32 ms one-token cycle, more
   than 2.2 accepted tokens per cycle would be required merely to clear the
   historical 68.2-token/s floor; the real requirement must be calculated from
   the measured multi-row cycle time.
4. Use joined Spark logs and one profiler trace to rank the remaining categories:
   Mac Metal work, MCDMA wake/copy time, Spark head/agreement work, and in-layer
   NCCL. Test the largest measured category first.
5. Test Mia's new two-twin-rail configuration as one isolated candidate. The
   second `f1` netdev and RoCE HCA are physically up on both Sparks but currently
   have no IPv4 GID. Give the twins a separate private subnet, prove RoCE-v2 GIDs
   and rollback, then compare against the one-rail baseline. Judge prompt and
   first-token shapes first; do not keep it merely because upstream reported a
   gain on different hardware state.
6. Measure a compiled Mac layer-0 forward candidate. Its theoretical ceiling is
   2.5-4% of a decode step; exact E1-E8 identity and offline activation equality
   are mandatory.
7. Reuse the already-validated frame digest and remove redundant prefill copies
   as one no-math-change candidate. Judge it on B3-2040; B1 may not regress 2%.
8. Test TensorFold EXL3-load/KDA settings only when the layer timer gives the
   category a measured ceiling above 3%. Keep `TF_GLM_L2PF=0`; this system
   already measured and rejected `1`, and chunked KDA is already active.
9. Test the MCDMA keepalive only if the idle soak proves a first-request wake
   penalty or measured transport overhead reaches `T`. First make the helper an
   owned adapter process visible to start/status/cleanup. Compare steady state
   and three 30-minute idle gaps, then stop it immediately.
10. Consider prompt windows above 64 only if B3 proves per-window fixed cost is
   material. This requires an explicit protocol and CUDA-buffer change.
11. Keep official TensorFold 0.6.3 isolated to the Mac MLX lane. Do not rebase
   Mia's 68 CUDA patches during this capacity proof. Any future unified-version
   candidate must pass the complete reference and recovery campaign.

MTP is mandatory in this campaign because the accepted comparison service uses
it and the one-token loop cannot meet the performance objective. General request
batching, concurrency, and streaming remain outside the single-user latency
campaign unless measurements show they are required for the final workload.
DFlash2 is explicitly allowed for this proof of concept. Record it in every
launch and receipt so a later production decision can replace or license it.
Piggybacking acknowledgements remains rejected because its measured ceiling is
below 1%.

## Soak, fault, stopping, and final gates

- Load soak: two hours of mixed requests, B1 every 15 minutes within `R`, with
  Mac API and both Spark memory sampled.
- Idle soak: eight hours idle, then exact canary plus B1 within `R`.
- Faults: worker killed mid-request, head killed idle, Mac API killed
  mid-request, Mac connector killed, Spark listener killed, control tunnel
  dropped, and client disconnect mid-request. Initial recovery budget is ten
  minutes per fault and becomes the measured proven budget afterward.
- Close an optimization category after two consecutive rejections or when its
  measured ceiling is below 3%. Stop when non-layer overhead is no more than 5%
  of a decode step, when the remaining open-category ceilings sum to less than
  3%, or immediately on a correctness/health failure.
- Final acceptance requires clean deploy, preflight, reference/equivalence
  suite, exact canaries, zero MCDMA failures/OOMs, both soaks, the fault matrix,
  repeated benchmark, clean stop/start cycles, committed receipts, canonical
  Linear updates, a clean repository, and a healthy three-machine service.

## Review record

Fable session `79945bde-174a-4d13-928c-8665d81cc589` rejected the earlier
five-run/3%-only rule, the low-entropy digit-only oracle, and transport-first
tuning. This revision uses restart-separated round medians, a counterbalanced
paired screen and confirmation, explicit tail/memory/idle/recovery gates, a
broader reference oracle, soaks, and a fault matrix. The read-only review JSON
has SHA-256 `bd957353d05ca1650662dcb209cf1bb2166816914ae03faa670a2d0deabb94fc`.
