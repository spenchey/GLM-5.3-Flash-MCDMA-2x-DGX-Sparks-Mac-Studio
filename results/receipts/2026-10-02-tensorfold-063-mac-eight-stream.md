# TensorFold 0.6.3 Mac eight-stream proof

Date: 2026-10-02

Result: **PASS**

Live upstream was rechecked on 2026-10-03. TensorFold `main` and the peeled
`v0.6.3` release still both resolve to
`9356df5c424b0c36b7737e37873a6f968b08de79`; no newer upstream release exists
in the repository. Mia's recipe and MCDMA also still match the commits pinned
in `UPSTREAM.lock`.

## Runtime

- Host class: Apple M3 Ultra with 256 GiB unified memory
- TensorFold: 0.6.3, official commit
  `9356df5c424b0c36b7737e37873a6f968b08de79`
- Prepared source tree:
  `72073174bce47f4f2f6e3f7b1f1824ae0e22e4fec46269c421334fa143baf573`
- Model: GLM-5.3-Flash MLX 4-bit, thinking disabled
- Server: loopback only, `max_batch_size: 8`, prompt cache disabled
- Workload: TensorFold's own `tools/bench_concurrent.py`, eight simultaneous
  requests, 256 output tokens, greedy decoding, three measured repetitions,
  and `--alone` token-hash checks

The measured server configuration was:

```text
python -m tensorfold serve <pinned-model-snapshot> \
  --host 127.0.0.1 --port 8898 \
  --name GLM-5.3-Flash-MLX-4bit \
  --parallel 8 --context 2051 --no-thinking \
  --mtp-drafts 3 --prompt-cache-gib 0 --snapshot-dir none
```

The raw process record preserves the resolved Python executable and exact model
snapshot path; the runtime record preserves the TensorFold source path, commit,
package version, and complete prepared-tree hash.

## Results

| Prompt | Aggregate tokens/s | Slowest first token | Exact vs solo |
| --- | ---: | ---: | ---: |
| Code | 87.9 | 1.43 s | 24/24 |
| Chat | 84.9 | 1.66 s | 24/24 |

There were zero failed and zero unmeasured replies. All 48 measured concurrent
replies were byte-equivalent at the token-hash level to their matching solo
run. The second prompt's steady shared-round rate was 89.1 tokens/s.

## Cleanup and evidence

The server stopped through `SIGTERM`, the loopback port closed, its `caffeinate`
helper stopped, and no TensorFold benchmark process remained. Raw benchmark
JSON, stdout/stderr, before/after health, memory pressure, process identity,
runtime versions, tool checksum, and cleanup proof are retained under
`results/raw/mac-tensorfold-063-eight-stream-20261002T1820Z/`.

This receipt proves the Mac's current eight-stream decode capability. It does
not by itself prove that the three-machine MCDMA service beats the current
two-Spark service; that is the separate fair-allocation gate.
