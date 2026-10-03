# Three-machine GLM inference receipt

Date: 2026-10-02

Result: **PASS — one real GLM-5.3 Flash inference path used the Mac Studio,
both DGX Sparks, TensorFold Metal/CUDA, and the real MCDMA boundary.**

## Pinned identity

- Mia recipe: `v1.3.2`, commit
  `92bf731c3aac61927726ef0c422b21b42111f2c4`.
- TensorFold recipe base: `0.6.0`; Mia patch label `ae8d1c789b47`.
- Identical Spark image ID:
  `sha256:0266e5767a4fb7cba90e587c6f69ab59dea24a0cd08ea983d333295717e9bf24`.
- Model revision:
  `9eaebb7c4e96d983dcd538e18624622ba5b820a8`.
- Mac stage file: 1,847,840,784 bytes; SHA-256
  `97210c3b3be7b0f2435157fd633fb2e8380d6fe8f923b5d7a698ecccc14e7235`.
- MCDMA implementation revision:
  `719219272c9ce6fc091b4eab6214eac510b5e387`.

## Compute proof

The Mac loaded the TensorFold tokenizer, embedding, and original GLM layer 0
through Metal. Each output was a contiguous BF16 `[rows,16384]` activation.
MCDMA delivered it to the head Spark. Both TensorFold CUDA ranks then executed
original layers 1 through 44, the head ran final norm and vocabulary head, and
both ranks agreed on each greedy token before the token returned to the Mac.

The prompt `Count from one to five:` used 6 prompt tokens and 15 generated
tokens. Across 21 committed positions it returned exactly:

The exact returned string was `" 1, 2, 3, 4, 5. "`.

The raw GLM-001 log SHA-256 is
`fb34e7b82e15a61e29da6f0e9561aa31d21f8c11a4e55adb037e5e4b192a5e1f`.

The persistent OpenAI-compatible endpoint then returned exactly
`THREE_MACHINE_READY`, with `finish_reason=stop`, 12 prompt tokens, 5 output
tokens, and the following measured inference split:

| Measure | Seconds |
| --- | ---: |
| End to end | 0.269177 |
| Mac Metal | 0.054137 |
| Spark plus MCDMA round trips | 0.210820 |
| Commit acknowledgements | 0.001995 |

FINAL-002 raw SHA-256:
`37bedda5220c5924560e7e0f8339d3b2af355e0e3ea7a4a01af616cea8344ba9`.

## Current health proof

FINAL-001 reran the full preflight and status checks after implementation:

- both Spark containers running;
- neither rank OOM-killed;
- identical pinned image IDs;
- MCDMA link up and idle between requests;
- both MCDMA peers alive;
- 1,572 calls on each side with zero failures; and
- persistent Mac API ready.

FINAL-001 raw SHA-256:
`a87273bc65e006dd841301c58afcaf4fce6638f9f070ae1df26f64d1790320f7`.

## State safety

Each activation produces a pending token. Metal and both CUDA ranks commit only
after a matching acknowledgement. Exact duplicates replay the pending token;
different, stale, corrupt, expired, or out-of-order frames fail closed. Reset
returns all three caches to position zero after every request.

## Scope boundary

This proves real three-machine greedy text inference. It does not claim MTP,
prefix reuse, sampled decoding, concurrency, streaming, tools, vision, or full
feature parity with Mia's two-Spark server. DFlash2 was not downloaded or used.
