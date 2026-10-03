# Mac GLM decode-capacity proof — 2026-10-02

## Question

Can the Mac Studio decode stage make the intended MCDMA design outperform the frozen native two-Spark GLM service?

## Frozen comparison

- Native two-Spark median decode: `71.823347204934 tokens/s`
- Required +3% target: `73.97804762108201 tokens/s`
- Prompt: `Output the integers from 1 through 2000 in order, separated only by commas. Do not explain and do not stop early.`
- Reply length: `128` tokens
- Expected output SHA-256: `bd541e7d43f21cd13484ec80ee2798a40e93e5312009084046a1b6c1faea169c`

## Mac service under test

- TensorFold: `0.6.2`
- Checkpoint: `Vontra/GLM-5.3-Flash-MLX-4bit-MTP`, snapshot `76add2a341a1cd90ad0e86bb69839ea9c35827c6`
- Resident weights: `168.5 GiB`
- Context: `2051`
- Built-in MTP: enabled, default depth `3`
- Listener: loopback-only `127.0.0.1:8898`
- Health after load: `status=ok`, `max_batch_size=8`

## Results

Every measured reply had the expected output SHA-256 above.

| Simultaneous replies | Median aggregate tokens/s | Change vs native | Clears +3% |
|---:|---:|---:|:---:|
| 1 | 62.4795 wall / 69.6133 engine decode | -13.0% / -3.1% | No |
| 2 | 72.3485 | +0.7% | No |
| 3 | 75.1731 | +4.7% | Yes |
| 4 | 77.4894 | +7.9% | Yes |
| 5 | 79.1377 | +10.2% | Yes |
| 6 | 78.1918 | +8.9% | Yes |
| 7 | 79.4425 | +10.6% | Yes |
| 8 | 80.0760 | +11.5% | Yes |

The three-run ranges were:

- width 3: `75.0327–75.2353 tokens/s`
- width 4: `77.3073–77.5473 tokens/s`
- width 5: `75.9635–79.4249 tokens/s`
- width 8: `79.9794–80.2653 tokens/s`

## MTP depth check

Five measured runs per depth showed the built-in default is the correct choice for this prompt:

| MTP depth | Median engine decode tokens/s |
|---:|---:|
| 2 | 68.1363 |
| 3 (default) | 69.6133 |
| 4 | 67.9360 |

## Conclusion

The intended Spark-prefill to Mac-decode design cannot beat the native service for one short reply because the Mac's single-stream decode is slower. It has a proven winning operating point for three or more concurrent replies: the Mac's shared decode rounds cross the +3% gate while preserving identical output. The production handoff must therefore optimize aggregate serving capacity and long-prompt latency, while reporting single-stream latency separately.

This receipt proves decode capacity only. It does not claim that GLM cache export/import over MCDMA is already implemented.
