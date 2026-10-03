# Upstream compatibility check

> Superseded later on 2026-10-02 by the pins in `UPSTREAM.lock` and
> `2026-10-02-tensorfold-063-mac-eight-stream.md`. This file preserves the
> earlier observation; it is not the current software claim.

Date: 2026-10-02

Result: **PASS — the deployed system uses the newest published compatible Mia
recipe and TensorFold base, while separately tracking the newer standalone
TensorFold release.**

Fresh upstream checks found:

- TensorFold `main` and release `v0.6.2` resolve to
  `56e2e3ec55bc0ae1d7d5158c4fa2c79a3567ab21`.
- Mia's recipe `main` resolves to
  `4bbf2f3d83f8b5363832368021c5b56b4ced3d92`.
- Mia's current changelog begins with `v1.3.2` and its published configuration
  still pins TensorFold `v0.6.0`, image tag `v0.6.0-ae8d1c789b47`, image digest
  `sha256:22789f0cb3dc308f0b2ce52a33961b88bd624af1725e91e8aba0a74a671bb969`,
  and model revision
  `9eaebb7c4e96d983dcd538e18624622ba5b820a8`.
- MCDMA `main` resolves to
  `e672c14ff9fc7b38994caf73025cf1588b4de74e`; this deployment remains pinned
  to the separately verified hardware revision
  `719219272c9ce6fc091b4eab6214eac510b5e387` rather than changing a working
  kernel/driver path during model integration.

Mia's newest owner change detects both PCIe halves of one cabled Spark QSFP
port. Live inspection proves `rocep1s0f1` and `roceP2p1s0f1` are up on both
Sparks, but only the first currently has an IPv4 RoCE-v2 GID. The second rail is
therefore an untested optimization candidate, not a current performance claim.

TensorFold 0.6.2 is not substituted into Mia's recipe because the recipe's 53
published patches and image are for 0.6.0. Moving to 0.6.2 requires a separate
patch rebase, full three-machine correctness run, recovery run, and performance
receipt; version recency alone is not compatibility proof.
