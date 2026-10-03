# Upstream refresh: TensorFold 0.6.1 and Mia recipe v1.3.2

Date: 2026-10-01

## Result

The newest compatible production pair is staged on both Sparks without
changing the running service:

- Mia recipe: `v1.3.2`, commit
  `92bf731c3aac61927726ef0c422b21b42111f2c4`.
- Recipe tree: `f1deb1301751339b160063ac875a471ec9f5d7c7`.
- Recipe TensorFold pin: `0.6.0`.
- Published image digest:
  `sha256:22789f0cb3dc308f0b2ce52a33961b88bd624af1725e91e8aba0a74a671bb969`.
- Recipe patch label: `ae8d1c789b47` with 53 patches.
- Model revision:
  `9eaebb7c4e96d983dcd538e18624622ba5b820a8`.
- Commercial drafter setting: `DRAFTER=mtp`; DFlash2 remains excluded.

TensorFold's newest standalone release is `0.6.1`, commit
`17c73e189f5e6a5304cda7ea37f086f9c49b4788`. It is not substituted into the
Mia recipe because Mia's published patches and image are for `0.6.0`.

## Evidence

| Check | Evidence | Result |
| --- | --- | --- |
| Official TensorFold head/tag | `main` and local exact tag resolve to `17c73e1` / `v0.6.1` | pass |
| Official Mia recipe head | `main` resolves to `92bf731`; changelog names `v1.3.2` | pass |
| Separate stage checkout | `/home/<spark-user>/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold-v1.3.2-stage` | pass |
| Shell syntax | every recipe `*.sh` passed `bash -n` | pass |
| Safe local overrides | worker `<spark-user>@<private-ip>`; MTP; one request; 1,048,576-token window; FP8 KV; q4 dense | pass |
| New recipe defaults | `TF_GLM_MULTI_LONE=0`; `TF_GLM_CACHE_ENTRIES=32` | pass |
| Published image | exact digest pulled; image reports `tensorfold 0.6.0` | pass |
| Both Sparks | image ID `sha256:0266e576...9bf24`, digest and patch label match on both nodes | pass |
| Model | existing 164 GiB snapshot verified by `tensorfold info`; worker has the same pinned revision | pass |
| Capacity | 1,342 GB free on head and 1,639 GB free on worker during preparation | pass |
| Service isolation | old `tensorfold-glm53:v0.5.0` containers stayed running, restart count 0, OOM false, health HTTP 200 | pass |
| Local checks | 20 scoped Python tests plus config, rollback and P07A contract tests | pass |

## Tracking

- Internal task `MOT-3373`: promote the prepared v1.3.2 pair.
- Internal task `MOT-3372`: track TensorFold 0.6.1 recipe compatibility.

## Boundary

The new pair is downloaded and prepared, not serving. Starting it needs the
existing planned maintenance gate because both full engines cannot coexist in
memory. The live 0.5.0 deployment remains the rollback target until that gate
passes. MCDMA integration remains separate and is not implied by this refresh.
