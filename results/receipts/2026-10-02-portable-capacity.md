# Portable serving-capacity receipt

Date: 2026-10-02 UTC

## Same-model two-Spark reference

The exact portable GLM checkpoint ran on both Sparks through TensorFold with
four simultaneous requests. One warmup and three measured rounds each returned
128 tokens per request and one stable output-token hash.

The measured aggregate rates were 121.63, 121.43, and 121.69 output tokens per
second. The median reference is 121.63 output tokens per second. The raw
`PORTABLE-NATIVE-002` log SHA-256 is
`def74c08b9cb31c4d0b8eac63d63a5b042d6601f5e62ad4fbf79ae159f611d8f`.

This is the required comparator. The older EXL3 result used a different
checkpoint and request shape and cannot decide whether MCDMA improves this
system.

## First cache-handoff attempt

The first full proof stopped during Spark prompt export. The live TensorFold
CUDA convolution cache has two dimensions for this unbatched prompt. The
connector only accepted the three-dimensional form used by its earlier unit
fixture. No performance result was produced.

The failed `PORTABLE-HANDOFF-001` raw log SHA-256 is
`871694cc011f1962f44bc4712638028a27142596e17a19e96085169aa41dd15f`.
The failed run's head log SHA-256 is
`83beae1fc51d0b7e202a96a95f9fc5f75495d8c6ecb5a473c5b0f7008c917191`.
The Mac decode process, both owned containers, and MCDMA processes were stopped;
the cleanup used no forced MCDMA kill.

The merge now accepts the live two-dimensional form and the earlier batched
form while preserving the required global q/k/v head order. Automated coverage
includes both shapes.

The repeated proof then reached the latent attention cache on both ranks and
stopped because TensorFold's NCCL wrapper does not define a collective for its
raw `uint8` FP8 cache storage. The `PORTABLE-HANDOFF-002` raw log SHA-256 is
`98b1490957f54523a2dca39096d4c0e8794677f89c894871298b8e0150f51011`.
The prompt exporter now uses TensorFold's supported bfloat16 representation,
which is also the representation the Mac decoder consumes. The optimized
native two-Spark reference remains on its FP8 cache setting.

The next attempt completed the actual three-machine computation and returned
the same token hash as both the Mac-local reference and the two-Spark
reference. Its one observed batch decoded at 92.90 aggregate tokens per second;
including 4.06 seconds of cache transfer, the full pipeline delivered 44.38
aggregate tokens per second. TensorFold wrote a tuning notice before the JSON,
so the proof parser correctly rejected the polluted artifact. The failed
`PORTABLE-HANDOFF-003` raw log SHA-256 is
`23957403c01d1e0dff3e4d437f5a149852360105026b3ffeea1afd3fc17c9794`.

Machine-readable decode now sends TensorFold notices to stderr and emits one
JSON record on stdout. Orderly shutdown now waits until MCDMA has accepted its
final acknowledgement before the Spark service disconnects.

## Acceptance gate

The repaired full path must return the same tokens in every round, prove both
Sparks and MCDMA participated, and exceed 125.28 aggregate output tokens per
second, which is 3% above the measured two-Spark median.

## Repaired proof and repeated result

The repaired four-request proof passed. Both Spark ranks prepared the prompt,
MCDMA transferred 592,052,224 bytes of prompt state, and the Mac returned the
same token SHA-256 as the two-Spark reference. MCDMA reported zero failures.
The proof decoded at 92.97 aggregate tokens per second and completed the full
path at 44.11 aggregate tokens per second. The proof result SHA-256 is
`53cac18ecb950d99be42607cac638dc3224d4e9eca41c7a8d9d8fa376c1ddd2d`.

The final benchmark loaded the Mac model once, ran one warmup, and then ran
three measured four-request rounds. Every reply matched the reference token
hash. MCDMA completed 1,856 calls per side during the benchmark with zero
failures. The full-path rates were 53.46, 51.05, and 52.64 tokens per second;
the median was 52.64. Mac-only batch decode rates were 92.86, 92.87, and 92.89
tokens per second; the median was 92.87. The benchmark JSON SHA-256 is
`f60bc9c23a4cd36ac8753ea15ccdb5d3be2960f769d15fb7e600ccbe9c18ee6b`.

The acceptance comparison failed with a measured gain of -56.72%. This is an
architectural result: even before MCDMA transfer cost, Mac decode at 92.87
tokens per second is below the same-model two-Spark result of 121.63. Reducing
transfer time cannot raise this exclusive split above the two-Spark reference.
The next design must use Spark and Mac decode capacity concurrently, or select
a model for which Mac decode itself exceeds the two-Spark reference. After the
measurement, both Spark ranks and all owned MCDMA processes stopped cleanly.
