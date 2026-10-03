# Three-machine performance and optimization receipt

Date: 2026-10-02

Result: **PROVISIONAL — useful pre-instrumentation observation, not a final
performance pass or optimization acceptance.**

This run predates the native token-reference gate, synchronized Spark phase
timers, rank-1 commit evidence, deployed identity enforcement, drift baselines,
and interleaved A/B comparison. It establishes only an older runtime
observation and one rejected setting. It must not be used to claim that the
current implementation is optimized or accepted.

## Accepted structural optimization

The OpenAI-compatible endpoint keeps the 1.8 GB Mac stage and tokenizer loaded
between requests. A warm exact five-token request completed in 0.282 seconds.
This removes repeated stage construction and weight loading from every request.
It does not change model math.

Warm API raw SHA-256:
`3b997be4eda0b745fda8bea49841d64de2050f779a86857824b4a3a87b0b2adf`.

## Fixed benchmark

- Prompt: `Write the numbers 1 through 100, separated only by commas.`
- Output budget: 128 tokens.
- Warmups: 1.
- Measured repeats: 5.
- Greedy output SHA-256 in every accepted and candidate run:
  `fe1fdf5d49caa68c1ef2f8a341413cb2c3826023aef1113c6c289dcd86193b32`.

| Configuration | Median tokens/s | Median inference seconds | Decision |
| --- | ---: | ---: | --- |
| Persistent baseline | 31.1568 | 4.1083 | baseline |
| TensorFold L2 bulk prefetch | 30.4497 | 4.2037 | reject, 2.27% slower |
| Restored accepted setting | 30.8898 | 4.1438 | keep |

Raw SHA-256 values:

- baseline: `aa1f27a5b3625c82ac25d67ba521be199d3e0f50d7671730ec0d0aff78746fd5`;
- rejected candidate: `52d3e5c084e17ae21eed64cbdc1d82b56f6fcf5f6f56127c83b92aa6a62993db`;
- restored confirmation: `841368d8cd0c5f2cd89a0e2e59c24d1340682a8a16be403dde66437204e47879`.

## Timing interpretation

In the restored five-run result, the median request spent about 0.30 seconds in
Metal layer 0, about 3.81 seconds in the Spark/MCDMA activation round trips, and
about 0.03 seconds in commit acknowledgements. The next useful optimization
area is therefore the repeated Spark-side round trip, not the Mac layer.

No speedup over the old two-Spark service is claimed: the architectures and
feature sets differ. This receipt measures the three-machine target against
itself and keeps only a correctness-preserving configuration.
