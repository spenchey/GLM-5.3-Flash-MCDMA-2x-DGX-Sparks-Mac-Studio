# Fable adversarial performance review

Date: 2026-10-02

Reviewer: Claude Code model `claude-fable-5-1`, maximum reasoning effort,
read-only tools, tmux session `fable-performance-plan-review`.

The review completed successfully in session
`d10a7cb7-d455-4371-b835-5eda0326c751`. Its structured JSON has SHA-256
`2ce7841fb131fef2f384c33e703f8b892d520d7237c88d4c05412e692e3b6b0c`.

## Material conclusions

- Success is exact model behavior plus proven Mac, MCDMA, and both-Spark
  participation, stability, recovery, reproducibility, and clean ownership.
- For this single-user non-streaming API, good performance means low and stable
  decode latency, time to first token, realistic end-to-end latency, zero
  stalls/errors, flat memory, and bounded recovery time.
- The old five-run/3%-only rule did not measure day-to-day drift and was not a
  defensible acceptance gate.
- MCDMA acknowledgement measured under 1% of each decode step. Transport-first
  tuning therefore had no credible 3% steady-state ceiling.
- A digit-list output hash was too weak to protect math/kernel changes.
- The benchmark lacked deployed identity, MCDMA call deltas, container state,
  tail latency, first-token tests, soak tests, and fault recovery timing.
- Mia's two-Spark speed is not a valid target because it used MTP/speculative
  decoding while this verified split deliberately does not.

## Accepted response

`docs/UPSTREAM-OPTIMIZATION-PLAN.md` now defines the success gates, B1-B4 and
E1-E8 suites, drift-aware interleaved comparisons, instrumentation, soaks,
fault matrix, stopping rules, and the optimization order. Transport keepalive
is deferred unless an idle test proves a wake penalty. The implementation adds
per-step timing, deployment identity, MCDMA call accounting, and the Mia-derived
bounded greedy code probe before any tuning result can be accepted.

This review defines the plan. It is not evidence that deployment, benchmarks,
soaks, or fault tests passed; those require separate live receipts.

## Focused success and optimization follow-up

Spencer asked the reviewer to define what success and good performance mean,
identify the bottleneck and plausible ceilings, and prescribe how to drive the
system toward them. Fable session `79945bde-174a-4d13-928c-8665d81cc589`
completed with read-only access and maximum effort. Its structured JSON has
SHA-256
`bd957353d05ca1650662dcb209cf1bb2166816914ae03faa670a2d0deabb94fc`.

The follow-up rejected the existing campaign as not yet defensible. Its main
points are now incorporated into the plan:

- the approximately 31 token/s result is a blended, pre-instrumentation
  observation, not a final baseline;
- the approximately 30 ms Spark/MCDMA span must be split into CUDA layer/head,
  Spark-to-Spark communication, and MCDMA wake/copy time before tuning order is
  chosen;
- a good service has exact reference tokens, p99 decode no more than 1.3 times
  median, no material memory growth, bounded idle wake and recovery, and proven
  use of all three machines—not merely a headline token rate;
- candidates use restart-separated counterbalanced rounds and paired statistics
  rather than treating repeated requests in one process as independent;
- accept only gains whose lower confidence bound clears 3%, with every
  correctness, identity, participation, health, memory, and secondary metric
  still passing; and
- close a category after two rejections or a measured ceiling below 3%.

The review also identified remaining prerequisite work: frozen native token
references, B1-B4 prompts, joined Spark events, real memory sampling,
request-receipt TTFT, idle-wake and recovery durations, and a stronger offline
experiment evaluator. Until those exist and run live, no optimization is
accepted.
