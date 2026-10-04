# Upstream contribution plan

The public recipe and upstream pull requests solve different problems. Do not
send this whole repository as one pull request to TensorFold or MCDMA.

## Publish here

This repository owns the three-machine orchestration, pinned configuration,
cross-engine transaction protocol, lifecycle scripts, performance harness,
receipts, and operator documentation. It links to upstream source instead of
copying model weights or complete upstream trees.

## TensorFold

Submitted: [TensorFold PR #385](https://github.com/ashhart/TensorFold/pull/385)
adds GLM-5.3 Flash DFlash2 support on MLX. It contains the target-layer capture
interface, GLM embedding and affine-head handling, imported-cache priming, and
seven focused MLX tests. Those tests passed on the M3 Ultra Studio against
TensorFold 0.6.5. The pull request does not contain model weights or the
mixed-machine orchestration.

## MCDMA

Submitted: [MCDMA PR #15](https://github.com/ashhart/MCDMA/pull/15) accepts
bandwidth protocol-v3 endpoint descriptors and closes both relay inputs only
after forwarding the final `CONFIRMED` proof. This fixes the observed v3
shutdown wait while retaining v1 support. Its focused suite passed 14 tests and
15 subtests.

The retained safe-cleanup and exact-verification patches remain recipe evidence,
not upstream submissions. The exact-verification patch does not apply cleanly
to current MCDMA and still needs separation plus fresh offline and hardware
checks before it should be proposed.

The three-machine result itself is better shared first as a sanitized MCDMA
issue or discussion linking this recipe and its receipts. That gives the
maintainer useful hardware evidence without asking MCDMA to own TensorFold- and
model-specific orchestration.

## Mia recipe

Submitted: [Mia recipe PR #58](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/58)
adds an independent qualification note. It preserves Mia's published two-Spark
baseline, records that the current three-machine single-request candidate is
still slower end to end, and links the public evidence without changing Mia's
defaults, patches, images, or measurements.

## Order

1. This recipe was published and independently test-scanned.
2. The focused MCDMA protocol fix was submitted separately.
3. The focused TensorFold GLM DFlash2 support was submitted separately.
4. The honest three-machine qualification was submitted to Mia's recipe as
   documentation, including the current failure to beat its published baseline.

Every submission must retain the credits and licenses in `NOTICE` and
`CREDITS.md` and must not include private deployment history.
