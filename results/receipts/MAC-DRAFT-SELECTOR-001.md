# MAC-DRAFT-SELECTOR-001

Status: REJECTED-WITH-PROCESS-GAP, never a serving candidate

## Question

Can TensorFold CUDA's measured one-head-per-round selector exploit the Mac
checkpoint MTP head's complementary guesses without paying to run both heads
in every round?

## Frozen identities

- Integration base: `8e40cb5`
- TensorFold experiment source: `8debc4e`
- Target checkpoint revision: `76add2a341a1cd90ad0e86bb69839ea9c35827c6`
- DFlash2 revision: `bf582e4eacc1810f76656d1811693ff6c6737d2a`
- DFlash policy: depth 2, four-bit helper, edge 1.4, tau 1.5,
  asynchronous layers 0-4
- Selector: DFlash first, two exploration rounds per arm, 3% switch margin,
  six-observation window, an alternate-arm probe every eight rounds, and a
  three-round shortened recheck after a switch
- Frozen output token SHA-256:
  `9684efe04b82897eff43afe857da02ad43b3c0b38e0779d81bd1c74a012f9bb6`

## Result

The three 400-token runs returned the frozen output exactly. Their complete
times were 7.159981, 7.157712, and 7.174237 seconds. Their decode rates were
57.881, 57.899, and 57.764 tok/s. Each run selected DFlash for 149 rounds and
MTP for 29, accepted 221 draft tokens, and required 178 target rounds.

The median was 7.159981 seconds and 57.881 tok/s. Against the retained
DFlash-only tau-1.5 result of 6.855346 seconds and 60.540 tok/s, the selector
was 4.44% slower in complete time and 4.39% slower in decode rate.

## Decision

Reject the selector for serving and keep DFlash authoritative. The selector can
measure which head has paid recently, but it cannot know which head will be
correct on the next round. Its periodic MTP probes therefore add more work than
the occasional complementary hit saves. Do not tune its window or switch
constants without a new, genuinely predictive pre-round signal.

The isolated implementation and tests are retained for reproducibility. No
serving configuration changed, and the remote wrapper, Python process,
caffeinate helper, staged files, logs, and exit files were absent after the
completed run.

## Evidence limitation

The successful wrapper removed its temporary log before the task could copy it
into `results/raw/`. The exact three result records were captured in the task's
terminal output, but there is no raw-file SHA for the successful run. This is a
process failure and prevents using the result as positive acceptance evidence;
it does not weaken the safe reject decision.

Two pre-load launch failures were retained:

- `results/raw/20261004T-selector-001-launch-failure.log`, SHA-256
  `0b877eef175f9c6ab2f29d8a9430a1508de07dc4925ef2eca3c5eb30ab65836a`;
- `results/raw/20261004T-selector-002-launch-failure.log`, SHA-256
  `1e651ee7398351bbd6190c183a1dfff174957bb7335d02b920453d87a3c4c136`.

Future experiment wrappers must preserve the completed raw log locally before
removing remote temporary files.
