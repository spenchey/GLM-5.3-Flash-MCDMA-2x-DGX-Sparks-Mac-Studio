# MAC-DRAFT-SHADOW-001

Status: PASS-DIAGNOSTIC, not a serving candidate

## Question

Does the GLM checkpoint's one-token MTP head correctly predict any first draft
tokens that the active DFlash2 helper misses?

## Frozen identities

- Host: Mac Studio `spencers-Mac-Studio.local`
- Integration base before this experiment: `ef3ca6e`
- TensorFold experiment source: `8debc4e`
- Target checkpoint revision: `76add2a341a1cd90ad0e86bb69839ea9c35827c6`
- DFlash2 revision: `bf582e4eacc1810f76656d1811693ff6c6737d2a`
- DFlash policy: depth 2, four-bit helper, edge 1.4, tau 1.5, asynchronous layers 0-4
- Frozen output token SHA-256: `9684efe04b82897eff43afe857da02ad43b3c0b38e0779d81bd1c74a012f9bb6`

## Result

Three 400-token measured runs returned the frozen output exactly. Each run
observed the same 170 rounds:

- both MTP and DFlash first-token prediction correct: 127;
- only MTP correct: 11;
- only DFlash correct: 10;
- neither correct: 22.

The three decode rates were 57.999, 57.887, and 57.945 tok/s; median complete
time was 7.155423 seconds. Evaluating both heads every round is therefore too
expensive even though their errors are complementary.

## Decision

DFlash remains authoritative and no serving configuration changed. The next
bounded question is whether TensorFold CUDA's measured drafter-selection method
can be adapted to choose the useful MTP rounds without evaluating both heads.
If a cheap pre-round signal cannot beat the retained DFlash-only result, reject
the hybrid path.

## Evidence

- Raw log: `results/raw/20261004T032432Z-MAC-DRAFT-SHADOW-001.log`
- Raw-log SHA-256: `922f884114af482502754b28e074f2e7cf3cc8fabc80ce9da2ba2425fba1f510`
- Harness: `scripts/benchmark-mac-draft-complementarity.py`
- Shadow runtime: `experiments/three_machine/mac_draft_shadow.py`
- Repository tests after recording: 264 passed
- Remote wrapper, Python, and caffeinate processes were stopped; the two staged
  experiment files and temporary Studio log/exit files were removed.
