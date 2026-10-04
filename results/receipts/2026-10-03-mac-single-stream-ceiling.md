# Mac single-stream ceiling

Date: 2026-10-03

This receipt tests whether Mac-only GLM decode is fast enough for a
Spark-prefill/MCDMA/Mac-decode request to beat MiaAI-Lab v1.5's published
two-Spark result. It does not include cache export, transfer, or import, so it
is an optimistic ceiling for the complete three-machine path.

## Frozen comparison

- MiaAI-Lab C1 prose claim: 60.4 decode tokens/s and 0.170 s TTFT.
- Fixed response length: 400 tokens; decode timing uses the 399 tokens after
  the first token.
- Implied Mia complete time: 6.775960265 s.
- Required three-machine time for a 3% win: 6.572681457 s or less.
- Mac checkpoint revision:
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6`.
- TensorFold: 0.6.4.
- One warmup followed by five measured greedy runs for each candidate.

## Results

| Experiment | Intentional change | Median decode | Median TTFT | Median complete time | Result |
| --- | --- | ---: | ---: | ---: | --- |
| MACCEIL-004 | MTP depth 3 | 58.637 tok/s | 0.261260 s | 7.064571 s | reject |
| MACCEIL-005 | MTP depth 4 | 58.500 tok/s | 0.261164 s | 7.081780 s | reject |
| MACCEIL-006 | MTP depth 2 | 58.957 tok/s | 0.260660 s | 7.027988 s | reject |
| MACCEIL-007 | MTP depth 1 | 59.315 tok/s | 0.251765 s | 6.978022 s | best MTP depth |
| MACPROF-008 | MTP depth 1 plus phase profiler | 59.321 tok/s | 0.251528 s | 6.977686 s | diagnostic |
| MACCEIL-009 | MTP depth 1 plus suffix-copy proposer | 59.400 tok/s | 0.252055 s | 6.969625 s | reject |

Every measured run returned token SHA-256
`9684efe04b82897eff43afe857da02ad43b3c0b38e0779d81bd1c74a012f9bb6`
and text SHA-256
`0e6e6c3707ea2eda2f118d908d798eed5820eb1591f1aeea40ef7c4a0a6730aa`.

MACCEIL-009 remained 1.655% below Mia's claimed decode rate and 2.858% slower
than Mia's implied complete time. It missed the required 3% win threshold by
0.396944 s, or 6.039%. The copy proposer made zero proposals, so the small
difference from MACCEIL-007 is ordinary run variation rather than an observed
copy-draft benefit.

MACPROF-008 consistently measured approximately 26.2 ms constructing each
family round, 3.7 ms waiting for the GPU, and 0.09 ms after the read. The next
candidate must therefore target the GLM round-construction path. Transport
tuning cannot repair this compute-ceiling miss.

## Raw evidence

- MACCEIL-004: `results/raw/20261003T154149Z-MACCEIL-004.log`, SHA-256
  `5f458fd84130ae73de95f72f6cc44330de330af49e6b492ba75460838bb7959c`.
- MACCEIL-005: `results/raw/20261003T154920Z-MACCEIL-005.log`, SHA-256
  `f1800cd64b76af6ef3864f982894331d66b7239c7ab13b816e05f716b27752d9`.
- MACCEIL-006: `results/raw/20261003T155108Z-MACCEIL-006.log`, SHA-256
  `1f42ae684f5f6e3fe2ada45b6317fdf3c7d1baa6fef0f5d372e97a570d113ae5`.
- MACCEIL-007: `results/raw/20261003T155300Z-MACCEIL-007.log`, SHA-256
  `ed44cd50630a8436f40b34082566d69b3c0bc3dfe346d4e97bc7fce648fbfded`.
- MACPROF-008: `results/raw/20261003T155615Z-MACPROF-008.log`, SHA-256
  `982c35fb83a12ed5bfd2c35cb540235ae2a471f4e20b304f9f388be9f8e70520`.
- MACCEIL-009: `results/raw/20261003T160045Z-MACCEIL-009.log`, SHA-256
  `10a44c0500816e3cb0fcea4ad2093d9a7f6bc4684a8c3db0730ac4faba6bdc06`.

