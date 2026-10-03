# Implemented TensorFold and MCDMA design

## Machine ownership

| Work | Owner |
| --- | --- |
| OpenAI-compatible endpoint and tokenizer | Mac Studio |
| Token embedding and original GLM layer 0 | Mac Studio, TensorFold Metal |
| Original GLM layers 1 through 44 | Both Sparks, TensorFold CUDA |
| Final stream mean, norm, vocabulary head, greedy token | Spark head, agreed by both ranks |
| Activation transport and control | MCDMA between Mac and head Spark |
| Rank synchronization | TensorFold NCCL between the two Sparks |
| Layer cache | Local to the machine that owns the layer |

Both Spark ranks load the pinned full checkpoint, then use a partial TensorFold
view whose layer list begins at original layer 1. This reuses TensorFold's
normal GLM CUDA kernels, cache, final norm, vocabulary head, and token selection.
The Mac loads only embedding plus original layer 0 from a deterministic stage
checkpoint derived from the same pinned EXL3 snapshot.

## Per-position flow

1. The Mac tokenizes the request.
2. TensorFold Metal embeds up to 64 rows and runs original layer 0.
3. The Mac flattens the four residual streams to BF16 `[rows, 16384]`.
4. MCDMA moves the framed activation to the head Spark.
5. The head mirrors the activation to the worker through TensorFold's NCCL
   group; both ranks run layers 1 through 44.
6. The sharded head selects a greedy token and verifies both ranks agree.
7. The token returns to the Mac without committing any cache.
8. The Mac sends a matching acknowledgement; only then do Metal and both CUDA
   ranks commit the same row count.

The two-phase commit is the protection against one machine advancing while
another loses a reply.

## Wire contract

- Magic: `TFMCDMA1`; protocol version 2.
- Fixed header: 104 bytes.
- Activation: BF16 `[1..64, 16384]`, row-major.
- One row: 32 KiB; maximum payload: 2 MiB.
- Identity: request ID, step, committed position, MCDMA generation.
- Integrity: payload length and SHA-256 over the complete header (with the
  digest field zeroed) plus payload. Request identity, shape, generation, and
  deadline therefore cannot be changed without detection.
- Liveness: absolute deadline on every frame.
- Control: reset, acknowledgement, shutdown, and bounded UTF-8 error.

The parser rejects malformed headers, corrupt payloads, extra bytes, expired
deadlines (including zero), invalid shapes, stale generations, another active
request, out-of-order positions, and a different retry while a token awaits
commit. An exact duplicate activation receives the cached pending token and is
not computed twice. Recently retired request IDs remain rejected after reset so
a delayed frame cannot reopen a completed transaction in the same generation.

## Lifecycle

The service starts MCDMA first, worker rank second, head rank third, and the Mac
API last. Readiness requires all four components. Normal shutdown reverses the
ownership path: API signal, model shutdown over MCDMA, clean rank exits, then
MCDMA cleanup.

Startup verifies image identity, patch label, complete model and Mac-stage
manifests, complete prepared TensorFold source, Mac Python dependencies, both
MCDMA libraries, and one deployed code/configuration identity on all three
hosts. It refuses to replace a running legacy service or a one-rank partial
state.

## Current limits

The layer-by-layer service is a verified proof path, not the performance
candidate or a drop-in copy of Mia's complete server. The current capacity path
runs Mia's patched TensorFold 0.6.0 CUDA server on both Sparks and official
TensorFold 0.6.3 shared-round Metal decode on the Mac. One immutable prompt cache
crosses MCDMA once and seeds up to eight identical-prefix Mac lanes through
TensorFold's cache copier. Arbitrary live prefixes, sampling, vision, tools, and
structured output remain separate changes. DFlash2 may be tested only as an
explicit proof-of-concept candidate.

The capacity proof compares sixteen concurrent requests in both cases: all
sixteen queued through a fresh pinned same-checkpoint two-Spark service versus
eight on that exact unchanged service plus eight on the Mac. The claim is
accepted only from counterbalanced paired rounds with an output oracle frozen
before the campaign, one common clock, exact three-frame transfer evidence,
and per-round container identity, configuration, restart, OOM, and health proof.

Mia's latest compatible image has 68 patches on TensorFold 0.6.0. Official
TensorFold 0.6.3 is isolated to the Mac stage. Mia's CUDA patch set is not mixed
into the newer base.

## Proof boundary

Accepted proof requires all three machine roles, nonzero MCDMA counters during
real model work, identical Spark image identity, zero link failures, an exact
answer, clean recovery, deterministic repeated output, and retained raw-log
hashes. A loaded driver, link light, ping, synthetic copy, or two-Spark answer
alone does not meet that boundary.
