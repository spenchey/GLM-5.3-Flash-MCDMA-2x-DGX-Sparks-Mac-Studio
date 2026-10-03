# Upstream contribution receipt

The useful contribution is a reproducible hardware result, not a claim that a
link light or ping worked.

## Include

- Mac, Spark, NIC, enclosure, cable, operating-system, and driver versions.
- Exact Git commits for MiaAI, TensorFold, MCDMA, oMLX, and any tested PR.
- Negotiated link speed, MTU, RDMA device names, and daemon versions.
- Byte-checked probe result and its timestamp.
- Model, quantization, context, cache format, prompt hash, and launch command.
- Container health, answer correctness, first-token time, token rate, and link
  counters before and after the request.
- The exact transport selection and proof that no silent fallback occurred.
- Every failure and manual patch, even when the final run passes.

## Do not include

- Passwords, tokens, private SSH configuration, public/private IP addresses,
  MAC addresses, GIDs, or user home paths.
- Model weights or third-party source copied into this repository.
- DFlash2; its license is not suitable for this commercial deployment.
- A claim of integration based only on a live interface, daemon, or synthetic
  memory transfer.

## Acceptance gates

1. The pinned artifacts and all three machine roles pass preflight.
2. Three clean stop/start/answer cycles pass through MCDMA.
3. The MCDMA counters move during each model request with zero failures.
4. Five identical greedy requests have one output hash and recorded timing.
5. Protocol corruption, ordering, duplicate, timeout, and lifecycle tests pass.
6. Raw logs are stored outside Git; the public receipt contains sanitized
   summaries plus SHA-256 hashes of the raw files.
7. Known API and feature limits are stated; a small compatibility endpoint is
   not presented as the complete upstream server.

Results belong under `results/receipts/`. `results/raw/` is ignored by Git.
Only a reviewed receipt should be posted to an upstream issue or pull request.

Before running anything, add an ID and hypothesis to
`docs/EXPERIMENT-LOG.md`. After the run, append the result and evidence; never
rewrite a failure into a pass. If a later finding changes the diagnosis, add a
correction that points back to the original ID.
