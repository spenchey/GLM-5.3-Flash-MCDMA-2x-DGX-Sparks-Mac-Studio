# Cache-handoff transport receipt

Date: 2026-10-02 UTC

## Outcome

The three-machine connector now uses MCDMA's documented pull mode: the two
Sparks stage the completed cache and the Mac Studio retrieves it with RDMA
READ. This matches the direction used in MCDMA's published split-inference
experiment.

The accepted project revision is `d882b02`. The MCDMA source pin remains
`e672c14ff9fc7b38994caf73025cf1588b4de74e`; no daemon binary or hardware
setting was changed.

## Preserved failure

The first recorded pull-mode check timed out even though the test service had
written a valid reply. Code and live logs showed that the one-shot validator
closed its service socket immediately after staging the reply. The listener
checks socket closure before it scans the staged word, so it could detach the
service and discard the reply. This was a validator race, not proof of an RDMA
failure. Its raw log SHA-256 is
`30b21886c9d71a033531f79b2a80be48345aca50014b0643f112c0dbdbfaf4e4`.

The validator now waits until the listener publishes the accepted reply word
before closing. The production CUDA prefill service is persistent and does not
use the one-shot lifetime that exposed this race.

## Repeated live proof

`CACHE-HANDOFF-PULL-002` ran three independent cycles. Every cycle proved:

- exact request SHA-256
  `66c60468130441e8b45cdf0e7e79ae9e74c08bf3c84d05e4d40ded131e363b9c`;
- exact reply SHA-256
  `cf7ef6b9110e2389d0140a03d32bf910de7a02b6ea8f39283cc237eed1776096`;
- one completed call and zero failures on both MCDMA daemons;
- reported reply transport `rdma-read-pull`; and
- orderly cleanup with no surviving child and no forced kill.

The raw log SHA-256 is
`ef673c3c4c3151970a06608a9d5b3408fc3ff9382c83beae8b26c72a2c7a4789`.

## Remaining gate

This proves the bidirectional transport and cleanup, not model performance.
The fair decision still requires the same portable GLM checkpoint and workload
on the two-Spark baseline and the full two-Spark-prefill/MCDMA/Mac-decode path.
The hybrid is accepted only if its median complete-pipeline rate is at least 3%
above the two-Spark median with identical output.
