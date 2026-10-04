# Spark prefill layer-readiness bound

Experiment: `SPARK-PREFILL-LAYER-TIMELINE-001`

UTC run: `2026-10-04T04:03:28Z`

## Question

Does the frozen Mia C1 prompt produce enough final cache state before Spark
prefill ends to justify a progressive MCDMA handoff, and can that change alone
clear the 6.572681-second complete-request target?

## Qualification

The CUDA instrumentation records events without synchronizing between layers;
one synchronization resolves the complete timeline after prefill. The frozen
33-token prompt fits in one prompt chunk. TensorFold's GLM CUDA KDA layer
commits its recurrent state inside each prefill layer, and DSA writes its cache
inside the layer. Therefore the event for a layer in this one final chunk is a
real cache-readiness boundary. For a multi-chunk prompt, only events from the
final prompt chunk have that property.

The pipeline result below is an optimistic upper bound, not a serving result.
It divides each measured gather, archive, wire, assembly, and verification
stage in proportion to each layer's bytes and assumes perfect overlap with no
per-piece scheduling cost.

## Frozen setup

- Project commit: `c4e1fb0`
- TensorFold source: `8debc4eb4df70650c2bf998b7728d039e8e4aaeb`
- Mode: full cold, positively attested DFlash2, five measured requests
- Prompt: Mia C1 prose fixture, 33 prompt tokens, 400 completion tokens
- Spark ranks: 2
- Cache: 45 layer groups, 81 tensors, 148,032,512 bytes
- MCDMA: one 160 MiB reply frame

## Measured result

The first request includes CUDA compilation and took about 1.595 seconds in
Spark prefill. Steady medians below exclude only that first compilation path.

| Spark-head stage | Steady median |
| --- | ---: |
| Prefill | 0.187586 s |
| Layer compute inside prefill | 0.179057 s |
| KDA cache collection | 0.044502 s |
| DSA cache collection | 0.001228 s |
| Archive build | 0.066599 s |
| Complete exporter | 0.300888 s |

| Mac handoff stage | Steady median |
| --- | ---: |
| Prepare request | 0.302886 s |
| Frame request | 0.097111 s |
| Frame assembly | 0.059583 s |
| Verification | 0.048643 s |
| Release | 0.000272 s |
| Complete handoff | 0.510967 s |

Layer-owned cache is 147,991,552 bytes. The final hidden state and MTP row add
40,960 bytes after the layers.

| Layer payload ready | Last included layer | Time from layer compute start |
| ---: | ---: | ---: |
| First layer | 0 | 1.152 ms |
| 25% | 10 | 44.962 ms |
| 50% | 22 | 89.817 ms |
| 75% | 33 | 131.826 ms |
| 90% | 40 | 160.398 ms |
| 99% / all layer bytes | 44 | 177.850 ms |

The byte-proportional five-stage model completes at 0.187674 seconds instead
of the current 0.510967-second handoff, an optimistic maximum saving of
0.323293 seconds. Applied to the measured 7.047043-second complete request,
that yields 6.723749 seconds, still 0.151068 seconds slower than the required
6.572681 seconds.

At the optimistic 0.187674-second first-token floor, the remaining 399 token
intervals need about 62.490 tokens/s. The measured decode median is 61.318
tokens/s, so progressive streaming still needs at least a further 1.91% Mac
decode improvement before safety margin. It also does not repair the exact
output mismatch at token 104.

## End-to-end acceptance result

- Median complete request: 7.047043 s — fail
- Required complete request: 6.572681 s — fail
- Median TTFT: 0.518408 s — fail
- Maximum complete request: 9.770920 s — tail fail
- Median decode: 61.317770 tokens/s
- First local-reference difference: token 104 — exactness fail
- MCDMA failures: zero — pass

## Evidence

Directory:
`results/raw/mia-c1-three-machine-cold-dflash-20261004T040328Z-72823/`

- `head.log`: `31888ad04102d11d98022865487d5b4ffa1a87cba9bd4875775a488d687e4a85`
- `worker.log`: `f776a2f988abefa095db979b6a8d9db21181defb430cabd1cad2eed5392c552c`
- `result.json`: `2eae9855c3c952130a1692c18461d98d230eea454365d623d7b2f0504c5d8819`
- `summary.json`: `67abbcf34587c6a89c9f516b3d775cb35831c6c6e0f4ef151990c11109291079`

## Decision

Proceed with a progressive cache-handoff protocol because the measured overlap
is material. Do not claim it can satisfy the goal alone. The accepted design
must publish only final-chunk layer state, retain complete tensor hashes and a
final seal before decode, and be combined with a measured Mac target-forward
improvement of at least about 2% plus the token-104 exactness repair. The
first-request CUDA compilation path must also be removed from the served tail
or completed during explicit startup qualification.
