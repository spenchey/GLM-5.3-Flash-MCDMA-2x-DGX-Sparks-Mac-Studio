# P07A: one complete TensorFold Mac layer

This isolated fixture loads exactly the embedding and base layer 0. It calls
TensorFold's own quantized weights, hyper-connections, KDA, and dense MLP. The
Mac runs TensorFold; oMLX is not imported. The serving system is not changed.

`window.py` exposes `forward_window(tokens)`, `commit(keep)`, and `reset()`.
The result is evaluated and synchronized contiguous BF16 `[rows,16384]`,
containing all four residual streams. It does not average streams or load the
final norm, vocabulary head, tokenizer, other layers, or MTP weights.

Every forward saves an independent bit-preserving entry-state copy before
MLX can mutate recurrent state. A partial commit restores that copy and
replays the accepted prefix through the same complete layer adapter. A zero
commit restores it directly. Full commit retains the evaluated exit state.
The retained state is three convolution rows, one recurrent matrix, and the
row offset. The caller must commit or reset before its next forward.

## Reproduce on the Studio

Use a private virtual environment outside the checkout. The validated
environment used Homebrew Python 3.13.15 with MLX 0.32.3,
mlx-metal 0.32.3, mlx-lm 0.31.3, and NumPy 2.5.3. No system Python was changed.
The declared TensorFold dependency ranges are checked before each run.

Copy these test files and the extracted P00 source to a private working
directory on the Studio. Preserve their relative paths. Copy the P00 source
manifest alongside them. Do not copy or redownload the existing checkpoint.
Then run:

```sh
bash scripts/run-p07a.sh \
  /path/to/private-venv/bin/python \
  /path/to/private-working-copy \
  /path/to/pinned-tensorfold-source.sha256 \
  /path/to/pinned-model-snapshot
```

The wrapper verifies the pinned source manifest and every listed source file,
enforces offline Hugging Face use, and runs two separate fixture processes.
It requires identical output hashes, state hashes, identities and comparison
results. Each child exits before the next starts. The fixture opens no server
or network socket and writes no checkpoint or tensor scratch files.

`tests/test-p07a.sh` checks syntax, the 53-tensor identity contract and the
absence of forbidden final-model calls without requiring a model or MLX.
The real-device fixture checks the actual checkpoint headers/config/index,
hashes exactly the 53 selected tensors' stored payloads at their safetensors
offsets, and records their individual hashes plus a combined digest. The
combined digest is SHA-256 of the UTF-8 canonical JSON name-to-hash dictionary
(sorted keys, separators `,` and `:`). No unselected tensor bodies are read
by the hashing helper. Offline tests prove range boundaries, short-read
failure, and all 53 hash entries while excluding intervening unrelated bytes.
The fixture explicitly asserts batch/chunk row offsets match for every case.
It also checks
five window sizes, short decode versus wider prefill, exact repeatability,
batch versus one-row chunks, invalid inputs, reset, and six commit cases from
a nonzero entry state. It records measured memory, elapsed time and both cache
hashes. The source manifest, checkpoint metadata and selected payload hashes
identify the input; this test does not rehash the entire 169 GiB checkpoint.

## Limits

Short decode windows of 4 and 16 rows match one-row chunks bit for bit.
Wider prefill uses TensorFold's ordinary MLX path and differs numerically from
one-row decode. The observed differences are reported in the receipt; no
acceptance tolerance is invented. In particular, a partial wide-window
commit replays the accepted prefix using the execution path for its new
length. It must not be assumed to preserve the original wide-window prefix's
bits. Integration must account for that distinction.

P07A proves local complete-layer execution and state handling. It does not
prove CUDA parity, language-model quality, MCDMA model transport, three-host
serving, or a speed benefit. Those have separate project gates.
