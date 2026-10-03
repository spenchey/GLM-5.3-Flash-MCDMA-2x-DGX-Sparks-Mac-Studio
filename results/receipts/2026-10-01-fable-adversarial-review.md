# Claude Fable adversarial review receipt

Date: 2026-10-01
Experiment: `REVIEW-001`

## Result

PASS: a read-only external review completed successfully. Its verdict was
`GO ONLY AFTER SPECIFIC PROOF`.

This is a plan-review pass, not approval to deploy. No live service, model,
route, MCDMA process, or host configuration was changed by the review.

## Reviewer run

- Provider: Claude Code CLI
- Model: `claude-fable-5-1`
- Effort: `max` (the installed CLI's highest setting)
- Claude session: `523fb6d3-f1de-4c59-9e5f-9461ebb384fa`
- Dedicated tmux session: `glm-mcdma-fable-review`
- Tool mode: read-only `Read`, `Grep`, and `Glob`; no writes, MCP, or hooks
- Exit code: `0`
- Prompt SHA-256: `7185fa41f388983f36609cab1c6a4c100c09c08aa99f5c7545064c1ebed62871`
- Raw log: ignored `.state/fable-review-output/20261001T160803Z-REVIEW-001.log`
- Raw log SHA-256: `3d5edad8a166dc2e89e089c3921035c94119a9d71c20035763182ebd43d23d0d`
- Tmux session was gone after completion, confirming the review program closed.

## Post-review safety check

At `2026-10-01T16:35:05Z`, `scripts/status.sh` reported:

- head container: running, OOM false, restart count 0;
- worker container: running, OOM false, restart count 0;
- API model: `GLM-5.3-Flash-EXL3`, owned by TensorFold;
- Studio interface `mcrdma1`: up and running; and
- TensorFold/MCDMA backend: explicitly not implemented.

This proves the review and documentation work did not replace or relabel the
live two-Spark service.

## Accepted corrections

1. A second full GLM engine cannot safely coexist with the live one on 128 GB
   Sparks. Full-model work needs a rehearsed stop/test/restore window.
2. Existing RDMA proof used validation clients, not `mcdma-rpcd`. The actual
   daemon needs its own hardware echo, counter, shutdown, and cable-fault proof.
3. The boundary is four residual streams, `[rows, 16384]` bf16: 32 KiB per row
   and 2 MiB for 64 rows.
4. Both TCP and MCDMA must follow the daemon's Mac-initiated request/reply flow.
5. MLX/EXL3 numerical divergence and disk headroom must be measured before
   choosing unified MLX or a mixed checkpoint boundary.
6. Prefix reuse starts disabled. MTP waits for Mac row-window and partial-commit
   equivalence to serial execution.
7. A full CUDA loopback split through the real codec and Mac relay must pass
   before building the Metal stage.
8. The honest performance target is a pre-registered regression bound, not a
   promised speedup.

## Cross-checks before adoption

The reviewer examined upstream TensorFold commit
`bb4b4a35863af562fc4ccb2586300d8f94b5d6de`, but the live image reports
TensorFold `0.5.0`, patch label `cefe8bf45d07`, and is built by a recipe with 52
patches. The image had no Git metadata. Therefore upstream-base claims about
live context, FP8 cache behavior, request concurrency, cancellation, and exact
code seams were not copied into the plan as live facts. Gate 0 now requires
extracting and hashing the patched image source.

Current upstream status was also checked rather than assumed:

- oMLX PR 3869 is merged, but the PR states its code was not yet run end to end
  on ConnectX hardware: <https://github.com/jundot/omlx/pull/3869>
- PR 3870 remains open: <https://github.com/jundot/omlx/pull/3870>
- PR 3941 remains open: <https://github.com/jundot/omlx/pull/3941>

Those projects remain design references, not deployment proof.

## Next five actions

1. Snapshot and hash the exact live TensorFold files and re-check all seams.
2. Benchmark the unchanged live service over five fixed seeds.
3. Prove `mcdma-rpcd` on the actual link with 32 KiB and 2 MiB checksums.
4. Measure checkpoint divergence and disk headroom, then close D-010.
5. Rehearse stop/test/restore three times before any full-model experiment.

Action 5 is the first service-impacting step. Actions 1 through 4 are designed
to leave the live two-Spark service running.
