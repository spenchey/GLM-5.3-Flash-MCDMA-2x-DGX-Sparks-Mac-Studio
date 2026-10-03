# P07A1 selected tensor payload and offset proof

Date: 2026-10-01

Linear issue: MOT-3365

Result: **PASS — all 53 selected tensor payload hashes reproduce in two independent Mac processes; all batch/chunk offsets asserted; prior numeric behavior unchanged.**

## Review findings addressed

The earlier P07A fixture verified the pinned config/index and selected tensor
headers but did not hash their stored bytes. It also recorded batch/chunk
offset equality without asserting it. This follow-up closes those two gaps;
the original P07A receipt remains unchanged.

`identity.py` now validates each selected safetensors offset range against its
dtype/shape byte count and file size, seeks to the payload's absolute start,
and hashes precisely its length in bounded 1 MiB reads. Only embedding and
layer-0 payloads are read. All 53 names must be present in the resulting
per-tensor hash dictionary. Its deterministic combined digest is SHA-256 of
UTF-8 JSON of that name-to-hex-hash dictionary, sorted by name and serialized
with separators `,` and `:`.

`fixture.py` now explicitly asserts `offset_equal` in every 4-, 16-, 17- and
32-row batch-versus-single-row comparison, before accepting that case.

## Recorded evidence

- Tensor payloads hashed: **53**.
- Selected stored bytes hashed per process: **521,011,928**.
- Combined selected-payload SHA-256:
  `6d59f6c34a5d7fcd304e6e6f8fc5ff51f2f8a8e2061331fd726c9f68fe4e4135`.
- All 53 per-tensor digests are checked into
  `experiments/p07a/evidence-2026-10-01.json` under
  `evidence.checkpoint.payloads.sha256`.
- Previous whole stable-evidence digest:
  `a217663750027ce5cbaf3bf6c0dd95af1aec79da4871174f65f1262a88df68ef`.
- New whole stable-evidence digest (includes new payload data):
  `12edc32eb47b79f50b3d011ce3bbb0c48ba34cc73036652dd94b76b161e507e2`.

The complete fixture exited successfully in **two separate processes** on
`spencers-Mac-Studio.local`. Their complete deterministic evidence, including
every payload digest, was identical. Before refreshing the checked-in compact
evidence, a structural JSON equality check compared every preexisting evidence
field to the rerun, removing only the new payload section and the untracked
expanded tensor-shape section. It returned `true`. This proves every prior
window output/state hash, commit replay result, batch/chunk numeric comparison,
input, zero-state hash, version and execution path stayed unchanged. The old
whole-record digest was not incorrectly required after extending the record.

All config/index/source checks remain: checkpoint revision
`76add2a341a1cd90ad0e86bb69839ea9c35827c6`; TensorFold source manifest
`aacb31df395dbb4fed6bd01929b47e1d3d6040b953e8ce457d2fc984303c016c`;
MLX 0.32.3 and mlx-lm 0.31.3 in the existing private Python environment.
These hashes establish reproducible local input bytes. They are not a claim
that the entire 169 GiB checkpoint was rehashed.

## Tests and timing

`tests/test-p07a.sh` passes. Three added offline tests verify exact bounded
payload reads (including a range spanning two chunks), truncation/invalid-range
failure, and all 53 real tensor names mapped to tiny synthetic payloads. The
last test records every read range and proves intervening unselected bytes
were excluded; it independently verifies each digest and the combined digest.
The static fixture check now rejects absence of an offset assertion.

The complete real-device runs took **1.705801** and **1.174680 seconds**,
including selected-payload hashing. Both used 528,712,672 active MLX bytes and
peaked at 633,816,328 MLX bytes. Peak RSS was 733,347,840 and 733,986,816 bytes.
These are test timings, not serving-performance claims. The previously
documented wider-prefill numeric differences are unchanged; no tolerance was
introduced.

## Preservation and cleanup

- Both live Spark GLM containers were independently rechecked running, OOM
  false, restarts 0. The expected `GLM-5.3-Flash-EXL3` model remains present and
  the explicit HTTP check returned 200.
- No P07A fixture/repeat process remains on the Studio.
- No generated `__pycache__` or `.pyc` exists in the owned experiment area.
- No checkpoint writes/download, Spark changes, API startup, DFlash2,
  production TensorFold modification or system Python change occurred.
- `git diff --check` passes. Raw complete rerun evidence is retained privately
  at `.state/P07A/two-process-payload-evidence.json`.

P07B remains the separate CUDA parity task after its P03 maintenance gate.
