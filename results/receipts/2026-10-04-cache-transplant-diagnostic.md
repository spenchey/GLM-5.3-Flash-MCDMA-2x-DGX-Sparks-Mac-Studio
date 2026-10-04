# Cache transplant exactness diagnostic

Date: 2026-10-04

## Question

Can a small Mac-local portion of the transferred prompt cache restore exact
output while both Sparks still do the useful prefill work?

The test used deployment `5e09272`, Mia v1.5 Spark image
`sha256:e97db95dd4b3f9a5ebd6ddeafb8d7d422dda729cc01b33d7f5c6ecfcf9cf2aaf`,
patch label `9f73cca659a1`, the pinned portable checkpoint, protocol 4, two
96 MiB frames, and the fixed C1 400-token reference.

## Result

Status: **pass as a diagnostic; reject partial cache transplantation**.

One 148,032,512-byte cache crossed MCDMA. Nineteen fresh variants then replaced
selected imported cache objects with Mac-local equivalents and followed the
same reference tokens until the greedy target decision differed.

| Variant | First different decision |
| --- | ---: |
| Imported baseline | 104 |
| All 45 local layer caches, local draft | none through 399 |
| All 45 local layer caches, imported draft | none through 399 |
| Local KDA layers only | 83 |
| Local MLA layers only | 24 |
| Local draft only | 104 |
| All local except early KDA | 104 |
| All local except middle KDA | 130 |
| All local except late KDA | 298 |
| All local except early MLA | 104 |
| All local except middle MLA | 81 |
| All local except late MLA | 81 |

Every tested prompt-layer family and range contributes to eventual numerical
drift. The draft cache does not cause it. Recomputing only a few Mac layers is
not an exactness repair; recomputing all 45 would duplicate the full prefill and
remove the reason to use the Sparks.

MCDMA advanced from 0 to 5 calls on each side with zero failures. Both Spark
containers used the exact current image, reported `OOMKilled=false`, and exited
through an acknowledged shutdown. No owned process remained.

## Preserved process failure

The first launch supplied a benchmark-only drift flag without its required
benchmark mode. Argument validation stopped before inference, and cleanup
passed. That evidence remains at:

`results/raw/cache-transplant-diagnostic-20261004T085829Z-80514`

## Evidence

Successful raw directory:

`results/raw/cache-transplant-diagnostic-20261004T090513Z-83042`

- `result.json`: `68b8682350ec1fd10e362c3d4c65b5e2473f4e64c178a0d35d5c3a6aa9df24f6`
- `summary.json`: `a463e304ae7f2b56587a97c182c67fe8fc17e4584426321c87ddc04dc19a095d`
- `head.log`: `6399101b4a211dd9757176ebb47ebe5f897ef71ff5641aca0aa6d3798280e501`
- `worker.log`: `65477c5810bb15eb417d51e2d747e5f17bbe173b50622782b28638370e8c6369`
- shutdown evidence:
  `results/raw/stopped-cache-handoff-20261004T090819Z-83906`

## Decision

Close partial cache transplantation. The next exactness investigation must
compare CUDA and Metal prompt-kernel numerics at layer boundaries. Performance
work should separately target the Mac target forward; neither problem is fixed
by another MCDMA direction or cache-copy experiment.
