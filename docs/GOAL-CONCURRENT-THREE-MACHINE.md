# Goal: concurrent three-machine GLM serving

## Objective

Prove one reproducible GLM-5.3-Flash capacity service where both DGX Sparks
decode through TensorFold CUDA while the M3 Ultra decodes another shared round
through TensorFold Metal. MCDMA must carry real Spark-produced prompt state to
the Mac. The result must beat a fresh, pinned, same-checkpoint two-Spark
TensorFold service handling the same number of simultaneous requests.

The older layer-by-layer split remains correctness evidence, but it is not the
performance candidate: it kept one side waiting and reached only 52.64 output
tokens per second.

## Pinned current software

- Sparks: Mia recipe `cf28cc4`, published image digest `14f15591…`, TensorFold
  0.6.0 plus patch label `5e01f1bb74d8` (68 recipe patches).
- Mac: official TensorFold 0.6.3 commit `9356df5` plus the reviewed two-file
  dense-stage loader patch; patched source tree `72073174…`.
- MCDMA: the already hardware-proven pinned runtime in `UPSTREAM.lock`.
- Checkpoint: the exact portable GLM-5.3-Flash MLX 4-bit snapshot already
  present on all three machines.

The package versions intentionally differ. Mia has not rebased its CUDA recipe
onto 0.6.3; the Mac needs official 0.6.3's current shared-round engine.

## Gate A: Mac eight-stream capability

On the M3 Ultra, run TensorFold's own `bench_concurrent.py` with thinking off,
eight simultaneous requests, 256 tokens each, three measured repetitions, and
`--alone` exactness checks. Pass requires:

- TensorFold reports 0.6.3 and `max_batch_size: 8`;
- zero failed or unmeasured replies;
- every concurrent token hash equals its request run alone;
- all eight first tokens arrive within 2 seconds;
- raw results, health, memory, runtime identity, and clean shutdown are saved.

Gate A passed on 2026-10-02: code prompt 87.9 tok/s and 1.43 s maximum TTFT;
chat prompt 84.9 tok/s and 1.66 s maximum TTFT; 48/48 measured replies exactly
matched their solo output.

## Gate B: fair three-machine capacity comparison

Compare the same sixteen simultaneous, identical 256-token requests:

1. Baseline A: start one fresh pinned portable-checkpoint two-Spark service and
   send all sixteen requests to it. Its four active CUDA lanes queue the rest.
2. Candidate B: keep that exact service and container configuration unchanged,
   send eight requests to it, while the Mac serves eight from one verified
   prompt cache transferred once over MCDMA.
3. Warm A and B once, then measure the counterbalanced order `ABBABAAB` without
   restarting either service or reloading the Mac model between rounds.

The candidate passes only when every output hash equals the solo reference
frozen before the campaign, both Sparks and the Mac show work, baseline rounds add zero MCDMA
calls, and every candidate round adds exactly five calls per side (prepare,
three data frames, release) with zero failures. Every paired candidate
throughput must beat its baseline by at least 3%, and the conservative lower
confidence bound for the paired gain must also exceed 3%. Report client-visible
first-token and tail completion times; do not hide a latency regression behind
aggregate throughput. This is a frozen shared-prefix capacity claim, not a
claim about arbitrary unrelated prompts.

Every round is rejected if either Spark container ID, image, command,
environment, mounts, labels, start time, restart count, OOM state, or health
changes. Mac latency uses the same barrier clock as Spark latency and starts
before transfer, cache import, or engine construction. The three cache-frame
lengths and their exact total bytes are retained for every candidate round.
The one-time prefix preparation cost is disclosed separately and is not part
of the steady-state reused-prefix timing claim.

## Implementation rule

One immutable Spark prompt cache is transferred per Mac batch. TensorFold's
`LaneEngine.copy_single_cache` seeds the remaining Mac lanes by sharing whole
MLX arrays and detaching views. Re-sending the same roughly 148 MB cache `N`
times is rejected because it serializes avoidable transport.

This first gate deliberately uses one frozen prompt. It proves the useful
capacity ceiling before changing TensorFold's live server to export arbitrary
request prefixes.

## Experiment and stop rules

- AutoResearch changes one setting at a time and records every keep/reject
  result. OpenResearch is the Windows-side notebook, not a runtime dependency.
- Reject wrong tokens, OOM, lost service, MCDMA failure, stale software,
  recovery failure, or incomplete cleanup before considering speed.
- DFlash2 is allowed for this proof of concept and must be named in commands
  and receipts.
- Preserve old images and source stages for rollback. Never pipe an upstream
  installer into a verified environment or overwrite a working stage.
- Do not route production traffic until Gate B, restart recovery, repeatability,
  and clean shutdown all pass and the owner separately requests cutover.

## Final result

Gate B passed twice from clean starts on 2026-10-03. The two-Spark medians were
137.478 and 137.800 tokens/s. The matching three-machine medians were 173.095
and 172.968 tokens/s, gains of 25.90% and 25.71%. The conservative lower bounds
were 25.65% and 25.23%. Every one of eight measured pairs passed, every output
was exact, MCDMA calls and cache frames matched the frozen contract, and cleanup
left no owned process or container running. The retained receipt is
`results/receipts/2026-10-03-three-machine-concurrent-capacity.md`.
