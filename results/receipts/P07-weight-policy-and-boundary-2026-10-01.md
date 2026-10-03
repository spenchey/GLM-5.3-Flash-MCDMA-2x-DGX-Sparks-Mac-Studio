# P07 weight policy and Mac boundary

Date: 2026-10-01

Linear issue: `MOT-3331`

Result: **POLICY AND BOUNDARY SELECTED; FULL-LAYER NUMERIC GATE BLOCKED**

## Selection

- Checkpoint policy: use the same pinned MLX checkpoint on the Mac and both
  Sparks for the isolated three-machine experiment:
  `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` at
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6`.
- Do not mix that checkpoint with the live Spark EXL3 checkpoint across a
  model boundary. The live service remains on
  `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` at
  `9eaebb7c4e96d983dcd538e18624622ba5b820a8` until the later maintenance-window
  canary.
- Smallest useful Mac interval: embedding plus base layer `[0, 1)`. The handoff
  is after layer 0 and before layer 1. It is one contiguous
  `[rows, 16384]` BF16 buffer, logically four `[rows, 4096]` residual streams,
  or 32 KiB per row. The Mac must own layer 0's KDA state.

This is a selected experiment design, not permission to integrate it. A full
embedding-plus-layer-0 forward has not run on the Mac and the numeric gate is
therefore blocked.

## Observed checkpoint facts

The read-only header helper scanned all indexed safetensors without reading or
rewriting model bodies.

| Fact | MLX revision `76add2...` | EXL3 revision `9eaebb...` |
|---|---:|---:|
| Indexed tensors | 114,160 | 150,226 |
| Checkpoint dtypes | BF16 76,174; F32 291; U32 37,695 | BF16 1,327; F16 74,304; F32 291; I16 37,152; I32 37,152 |
| Layer 0 tensors | 50 | 26 |
| Layer 0 tensor bytes | 164,168,408 | 579,060,440 |
| Layer 0 shards | 2 | 1 |

Layer 0 has 26 shared semantic tensor names. Fourteen have the same shapes and
dtypes: the six hyper-connection tensors, two norms, `A_log`, `dt_bias`, three
convolution weights, and `o_norm`. The other 12 are matrix weights. EXL3 keeps
those dense-layer matrices as full BF16 tensors, while MLX represents each as
a packed U32 weight plus BF16 scales and biases. For example:

- `layers.0.self_attn.b_proj.weight` is BF16 `[64, 4096]` in EXL3.
- In MLX it is U32 `[64, 512]` plus BF16 scales and biases, each `[64, 64]`,
  with affine 4-bit groups of 64.

The formats therefore describe the same layer topology but not the same stored
numbers. TensorFold's GLM CUDA recipe declares both MLX 4-bit and the pinned
Mia EXL3 variant as supported inputs; that is structural loader support, not
cross-format equality.

## Deterministic numeric fixture

The fixture dequantized the entire MLX layer-0 `b_proj` matrix to float32 and
read the corresponding EXL3 BF16 matrix as float32. It selected 512 positions
with index `(i * 104729 + 17) % 262144`, then multiplied both matrices by the
same float32 input `x[j] = (((j * 17 + 3) % 101) - 50) / 50` using NumPy.

| Measurement | Result |
|---|---:|
| MLX dequantized matrix SHA-256 | `b203c7a3cd208dc2694f0abb18a5e2c0ea2070e37a2bb1cfb3ba463381a6822c` |
| EXL3 BF16 matrix SHA-256 | `f8fc527671952ff6d5e6e71107224646eb93b5e21b94cd29a6e28623ee3b0ff0` |
| 512-weight maximum absolute difference | 0.01318359375 |
| 512-weight mean absolute difference | 0.0026986132143065333 |
| 64-output maximum absolute difference | 0.33564436435699463 |
| 64-output mean absolute difference | 0.09111599484458566 |
| 64-output RMSE | 0.11881915248692092 |
| MLX projection SHA-256 | `2e00ba251f3cc3cdd3e7e7b003a5347f5731bca8afb701fb7b7146b6b1813da0` |
| EXL3 projection SHA-256 | `508f757c61ae801e421dd6eb912e984402030e6cebef039bea9f46d40844a446` |

These are observed differences, not an acceptance tolerance. No tolerance is
selected because this projection does not include hyper-connections, KDA
state, the dense MLP, or downstream accumulation.

## Smallest Mac candidate memory and execution

On `spencers-Mac-Studio.local`, the helper evaluated the 53 tensors for the
embedding plus layer 0 from the pinned MLX checkpoint and executed one real
MLX 4-bit Metal `quantized_matmul` for `b_proj`:

- MLX runtime used for this isolated fixture: `0.31.2`.
- Active MLX memory after evaluation: 521,536,216 bytes (497.376 MiB).
- Increase over the helper's pre-evaluation state: 520,864,472 bytes
  (496.735 MiB).
- Process peak RSS: 635,092,992 bytes (605.672 MiB).
- Header/tensor evaluation: 0.08390 seconds.
- Quantized projection forward: 0.00289 seconds, output shape `[64]`.

This proves that the smallest candidate's weights fit and that a constituent
quantized projection runs. It does **not** prove that the complete GLM layer
runs. The installed `mlx-vlm` 0.6.3 package and inspected oMLX rc2 source on the
Mac contain no `glm5_next` model implementation, so constructing a complete
layer would require new model runtime code or the exact previously validated
runtime source. That work is outside P07.

## Disk headroom

All values below were observed with `du -skL` and `df -Pk`.

| Host | Current checkpoint | Free | MLX checkpoint addition | Free after addition |
|---|---:|---:|---:|---:|
| Mac Studio | MLX already present, 169.260 GiB | 162.409 GiB | none | 162.409 GiB |
| Head Spark `<head-spark>` | EXL3 present, 163.649 GiB | 1,341.400 GiB | 169.260 GiB | 1,172.139 GiB |
| Worker Spark `<worker-spark>` | EXL3 present, 163.649 GiB | 1,638.741 GiB | 169.260 GiB | 1,469.481 GiB |

This proves disk capacity only. The MLX checkpoint was not downloaded to either
Spark in P07.

## Why `[0, 1)` is the smallest useful interval

Observed TensorFold code embeds token rows into the four-stream buffer and
runs attention plus feed-forward work as one `layer_forward` state transition.
Splitting inside it would expose hyper-connection scratch and KDA internals,
not the declared residual boundary. One complete base layer is therefore the
smallest boundary-compatible unit. Layer 0 is also one of the three small dense
layers: its MLX tensors total 164,168,408 bytes, while the first MoE layer
(layer 3) is 4,165,991,256 bytes.

The performance value of moving only layer 0 is an inference and remains
unknown. The interval is selected because it is the smallest correct test, not
because a speedup has been measured.

## Block and smallest next task

P07 does not establish cross-format whole-layer equality and does not invent a
tolerance. The smallest next task is **P07A: isolated unified-MLX layer-0
parity**:

1. locate and pin the exact GLM5 Next MLX runtime that previously loaded this
   revision, or add a minimal isolated layer adapter;
2. run embedding plus layer 0 on deterministic token rows on the Mac;
3. during an approved isolated Spark maintenance test, run the same pinned MLX
   bytes through TensorFold CUDA;
4. compare the complete post-layer-0 `[rows, 16384]` BF16 buffer and record the
   observed difference before P08 chooses any acceptance limit.

No DFlash2 artifact was downloaded. No model, container, service, network,
driver, or TensorFold runtime file was changed.

## Live preservation and cleanup

After the analysis:

- Head and worker containers: `running=true`, `oom=false`, restart count `0`.
- Head model API: HTTP `200`; model ID `GLM-5.3-Flash-EXL3` remained present.
- No model process was left running on the Mac; the fixture processes exited.
- No temporary checkpoint or conversion output was created.

## Reproduction helpers

- `scripts/analyze-glm-checkpoint.py` reads and validates safetensors headers.
- `scripts/p07-numeric-fixture.py` performs the isolated matrix fixture and
  Mac memory/projection check.
