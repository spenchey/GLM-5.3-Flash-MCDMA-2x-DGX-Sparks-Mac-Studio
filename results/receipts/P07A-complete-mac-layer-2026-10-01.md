# P07A complete Mac TensorFold layer

Date: 2026-10-01

Linear issue: MOT-3358

Result: **PASS — isolated complete embedding + layer [0,1), repeated in two independent processes. CUDA parity remains P07B.**

## Identity and scope

- Actual target: `spencers-Mac-Studio.local`.
- Engine: hash-matched TensorFold 0.5.0 extracted in P00, base
  `9cd52ab4daba68ddd09be89be8f23ad43175e821`; source-manifest SHA-256
  `aacb31df395dbb4fed6bd01929b47e1d3d6040b953e8ce457d2fc984303c016c`.
  The wrapper checked every file in that manifest before execution.
- Checkpoint: already-present `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, revision
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6`.
- `config.json` SHA-256:
  `f80682bcc8ba5fc6ecdc5ce553700b02befe999bc4a6ba6fd47f644df4864adf`.
- `model.safetensors.index.json` SHA-256:
  `6c030d5b58f4c264f00becf9dddcb8f8b1f69b3bbaaf35c93592b201b673208f`.
- All 53 selected tensor names, header dtypes and shapes exactly matched the
  explicit fixture contract. The audited loader allowed only these 53 names
  and proved that every one was requested.
- Private environment: existing Python 3.13.15; MLX/MLX-Metal 0.32.3;
  mlx-lm 0.31.3; NumPy 2.5.3. No system Python change.

The adapter calls TensorFold `Weights.q("embed_tokens")` and
`load_layer(..., 0)`. It follows TensorFold's HCK fused short-window path and
ordinary `Layer` wider-window path, stopping at the complete layer boundary.
It imports no oMLX code. No final stream average, final model norm, vocabulary
head, tokenizer, MTP, additional layer or API is constructed. The static test
rejects the forbidden conclusion calls and the loader rejects any unapproved
tensor request.

## Numeric and state proof

Inputs are the deterministic IDs `((i * 7919 + 17) % 154880)` for
`i = 0..31`; no tokenizer is needed. Every output is synchronized, contiguous
BF16 `[rows,16384]`, preserving four `[rows,4096]` streams.

| Rows | Bytes | Output SHA-256 |
| ---: | ---: | --- |
| 1 | 32768 | `5933a1d79c1cb054814bfe7dea42fe3a08b4f4222beb36936f8739c881d70b8f` |
| 4 | 131072 | `75bf28e858ddb3f7be44e3d783c4d426281a9cf64712c8b546f7e6ad14061954` |
| 16 | 524288 | `4fdc46af0c105fc34d71f84448589c89105cb579c4d897bd7ffb1b9dfad73aeb` |
| 17 | 557056 | `7b0737ea84f33b0605a96cf0f097c675a6e0c2453b34a78f2a267490f59dd5b8` |
| 32 | 1048576 | `c82680cd7d00898f570802dc132b19316c72a1ae2cca455ab707a42596744572` |

Each size ran twice after reset inside each of two separate processes.
All output and state hashes matched. The complete deterministic evidence
SHA-256 is `a217663750027ce5cbaf3bf6c0dd95af1aec79da4871174f65f1262a88df68ef`.
The checked-in compact evidence is
`experiments/p07a/evidence-2026-10-01.json`; the explicit 53-tensor contract is
in `identity.py`. Full raw JSON is retained privately at
`.state/P07A/two-process-evidence.json`.

Zero state was verified as BF16 convolution `[3,24576]`, FP32 recurrent state
`[1,64,128,128]`, and offset zero. Shapes/dtypes remain checked after every
forward. Commit tests used a nonzero 3-row entry, then 4- and 17-row windows,
keeping zero, half, and all rows. Accepted-prefix replay, both state hashes,
offset and next-token output matched exactly in all six cases. Entry caches
are copied through independently owned host storage before forwarding to
protect against MLX in-place mutation. Invalid tokens and commit lengths are
rejected without silently advancing accepted state.

## Batch versus chunks: measured limitation

The 4- and 16-row short paths matched repeated one-row calls exactly for
output, convolution state and recurrent state. Both fused HCK and fused KDA
were exercised. The 17- and 32-row wider paths used ordinary HCK/KDA code.

| Wider batch vs one-row chunks | 17 rows | 32 rows |
| --- | ---: | ---: |
| Output maximum absolute difference | 0.000732421875 | 0.000732421875 |
| Output RMSE | 0.00008473897469229996 | 0.00008405413245782256 |
| Convolution maximum absolute difference | 0.015625 | 0.015625 |
| Recurrent maximum absolute difference | 0.0005186796188354492 | 0.0005488842725753784 |

These observations reflect different upstream short/wide execution paths;
they are not an approved tolerance. A partial wide-window commit replays its
prefix using the prefix-length path, so its state matches fresh accepted-
prefix execution, not necessarily the original wide window's prefix bits.
P08 must account for this; P07B still must compare the complete CUDA layer.

## Memory and elapsed time

Both independent runs loaded 528,712,672 active MLX bytes and peaked at
633,816,328 MLX bytes. Peak process RSS was 733,134,848 and 733,560,832 bytes.
The full fixture (including loading, repetitions, chunk and replay tests)
took 0.976363 and 0.984179 seconds with warmed filesystem/kernel caches.
These are fixture timings, not serving throughput or an acceleration claim.

## Checks, preservation and cleanup

- `tests/test-p07a.sh`: PASS.
- Real two-process fixture: exit 0, identical deterministic evidence.
- `git diff --check`: PASS.
- Final independent live status: both Spark GLM containers running, OOM false,
  restart count 0; head model ID `GLM-5.3-Flash-EXL3`; explicit HTTP check 200.
- Final process check: no P07A fixture/repeat process remained on the Studio.
  Existing Figmatrace and unrelated desktop processes were preserved.
- No live Spark/config change, API startup, checkpoint download, DFlash2,
  model scratch file or retained model process. The private reproducible
  environment/source files remain on disk; both model processes exited and
  their memory was released.

This passes P07A only. It does not establish complete three-machine serving,
CUDA parity, end-to-end answer quality, failure recovery or performance gain.
