# Upstream contribution plan

The public recipe and upstream pull requests solve different problems. Do not
send this whole repository as one pull request to TensorFold or MCDMA.

## Publish here

This repository owns the three-machine orchestration, pinned configuration,
cross-engine transaction protocol, lifecycle scripts, performance harness,
receipts, and operator documentation. It links to upstream source instead of
copying model weights or complete upstream trees.

## TensorFold

Candidate contribution: the small Metal loader change that reads the tested
EXL3 checkpoint's dense projections while retaining quantized projections.

Do not open the pull request from the retained 0.6.3 patch as-is. TensorFold
0.6.4 is now current and the old patch does not apply cleanly. Rebase the idea
onto current `main`, add a failing-before/passing-after checkpoint test, then
attach TensorFold's required exactness, prompt-speed, decode-speed, platform,
and test receipt. Keep the mixed-machine orchestration out of that pull request.

## MCDMA

Candidate contribution 1: bounded benchmark cleanup and termination handling.
The retained `0001-safe-benchmark-cleanup.patch` applies cleanly to current
MCDMA `main` at `e672c14ff9fc7b38994caf73025cf1588b4de74e`. Run MCDMA's full
offline suite and submit this as one small pull request with the cleanup-failure
tests and measured boundary.

Candidate contribution 2: exact verification windows. The retained
`0002-exact-verification-windows.patch` does not apply cleanly to current
MCDMA and is not ready to submit. Rebase it, separate unrelated changes, and
rerun both offline and hardware checks before considering a second pull request.

The three-machine result itself is better shared first as a sanitized MCDMA
issue or discussion linking this recipe and its receipts. That gives the
maintainer useful hardware evidence without asking MCDMA to own TensorFold- and
model-specific orchestration.

## Order

1. Publish and independently clone-test this recipe.
2. Open the MCDMA cleanup pull request after its upstream suite passes.
3. Rebase and measure the TensorFold loader change on current `main`, then open
   its own pull request.
4. Share the complete three-machine result with both maintainers, linking the
   public receipts and clearly stating tested versions and limits.

Every submission must retain the credits and licenses in `NOTICE` and
`CREDITS.md` and must not include private deployment history.
