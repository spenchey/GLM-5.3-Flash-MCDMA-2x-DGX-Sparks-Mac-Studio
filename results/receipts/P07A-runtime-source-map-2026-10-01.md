# P07A runtime source map

Date: 2026-10-01

Linear issue: `MOT-3358`

Result: **SOURCE FOUND AND PINNED; COMPLETE LAYER RUN NOT YET STARTED**

The implementation source is the hash-matched TensorFold 0.5.0 package
extracted from the live Spark image, not the nearby research checkout and not
oMLX. Its source-manifest SHA-256 is
`aacb31df395dbb4fed6bd01929b47e1d3d6040b953e8ce457d2fc984303c016c`,
and its declared TensorFold base commit is
`9cd52ab4daba68ddd09be89be8f23ad43175e821`.

The extracted source already contains the required Mac/MLX path:

- `tensorfold/families/glm5_next/model.py`: embedding, four-stream hidden
  state, hyper-connections, and layer dispatch;
- `tensorfold/families/glm5_next/kda.py`: KDA projections, convolution,
  recurrence, and cache updates;
- `tensorfold/families/glm5_next/mlp.py`: dense layer-0 MLP;
- `tensorfold/families/glm5_next/linear.py`: MLX affine 4-bit execution;
- `tensorfold/families/glm5_next/weights.py`: exact checkpoint-name mapping;
- `tensorfold/families/glm5_next/caches.py`: convolution/SSM state;
- the GLM Metal kernels under `tensorfold/kernels/glm/flash/v1/`.

The selected checkpoint remains
`Vontra/GLM-5.3-Flash-MLX-4bit-MTP` revision
`76add2a341a1cd90ad0e86bb69839ea9c35827c6`. The adapter must load the
embedding and `load_layer(..., 0)` directly. It must not call `load_backbone`,
which also loads the final norm and vocabulary head, and it must not call the
normal `hidden_rows` conclusion, which averages the four streams and destroys
the required boundary.

The correct output is the post-layer-0 BF16 tensor `[rows,4,4096]`, evaluated,
synchronized, made contiguous, and viewed as exactly `[rows,16384]`, or
`rows * 32768` bytes. Layer 0 owns a BF16 convolution state `[3,24576]`, an
FP32 recurrent state `[1,64,128,128]`, and an integer row offset.

The minimum isolated interface is:

- `forward_window(tokens) -> contiguous BF16 [rows,16384]`;
- `commit(keep)`;
- `reset()`.

It must prove separate short (`rows <= 16`) and wider (`rows > 16`) execution
because TensorFold selects different fused/ordinary MLX paths. Tests must also
cover batch versus chunked input, reset reproducibility, zero/partial/full
commit replay, exact 53-tensor identity, state shapes, output byte count, and
the absence of stream averaging, final norm, vocabulary head, tokenizer, API,
or retained model process.

## Current execution blocker

The Mac's default Python reports MLX 0.32.1 and does not have the required
TensorFold/MLX-LM environment. TensorFold 0.5.0 declares MLX
`>=0.32.2,<0.32.4` and MLX-LM `>=0.31.3,<0.32`. P07A must create a private
environment and prove these exact versions without changing the system Python
or starting a service.

oMLX commit `87460f4d50de79aef9b67e99e215c31f0a89b445` remains a useful
Apache-2.0 reference for GLM5 Next checkpoint handling and MTP rollback. It is
not the P07A runtime and does not replace TensorFold.

No file on a live host was changed, no checkpoint was copied, and no model or
service process was started during this source-mapping step.
