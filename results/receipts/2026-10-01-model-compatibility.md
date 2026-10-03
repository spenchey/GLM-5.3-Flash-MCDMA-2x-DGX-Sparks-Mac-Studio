# Mac/Spark GLM checkpoint compatibility receipt

Date: 2026-10-01

## Result

PASS-WITH-OPEN-GATE: the checkpoints are structurally compatible enough to
build a boundary test. This does not yet prove that mixed MLX 4-bit and EXL3
layers produce acceptable numerical output.

## Mac Studio checkpoint

- Repository: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`
- Revision: `76add2a341a1cd90ad0e86bb69839ea9c35827c6`
- All 43 weight shards resolve to present files.
- Logical and allocated weight size: 169.23 GiB.
- Studio memory: 256 GB.
- Free storage after the checkpoint: 163 GiB.
- Embedding plus first 1 or 2 base layers touches 3 shards / 9.32 GiB.
- Embedding plus first 4 base layers touches 6 shards / 21.32 GiB.
- Embedding plus first 8 base layers touches 14 shards / 53.30 GiB.

## Spark checkpoint

- Repository: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
- Revision: `9eaebb7c4e96d983dcd538e18624622ba5b820a8`
- It remains pinned and complete on both Sparks.

## Exact structural comparison

- Normalized semantic-config SHA-256:
  `c0d14c3d2d5c3de6e4e3bfa8d9c56c55cb42252cf4c23ec23f6af4bf841e9135`
- Both declare `Glm5NextForConditionalGeneration`, hidden size 4096,
  45 base layers, one next-token-prediction layer, 64 attention heads, 288
  routed experts, eight experts per token, vocabulary 154880, and context
  1,048,576.
- `tokenizer.json` matches:
  `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`.
- `tokenizer_config.json` matches:
  `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc`.
- `generation_config.json` matches:
  `230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f`.
- The chat-template files differ. The target design keeps request tokenization
  and the API on the Spark head, so the Mac template must not be used.

## Remaining gate

The quantization formats differ. The next experiment must feed a fixed hidden
state through the proposed split, compare TCP and MCDMA byte paths, and measure
output/logit deviation against the unmodified two-Spark baseline. No final
compatibility claim is allowed before that result.
