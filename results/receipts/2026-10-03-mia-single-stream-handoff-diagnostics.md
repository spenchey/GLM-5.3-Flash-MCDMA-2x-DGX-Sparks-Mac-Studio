# Mia single-stream handoff diagnostics

Date: 2026-10-03

This receipt records the first fair three-machine comparison against
MiaAI-Lab v1.5's published two-Spark C1 prose result, the proven correctness
failure in the original cache boundary, and the local proof for the repaired
boundary. It does not claim that protocol v2 has passed on all three machines.

## External target

- Published two-Spark decode: 60.4 tokens/s.
- Published time to first token: 0.170 s.
- Fixed response: 400 generated tokens.
- Implied complete time: 6.775960265 s.
- Required complete time for a 3% win: 6.572681457 s or less.
- Prompt: `Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough.`
- Protocol: one 32-token warmup, then five greedy 400-token runs, thinking off.

These published numbers are the requested external performance target. No Mia
checkpoint download is needed to compare against them. The portable checkpoint
already present on all three hosts supplies only the exact-output control.

## First live handoff result

`MIA-HANDOFF-001` used both Sparks for prompt prefill, MCDMA for the cache
transfer, and the Mac Studio for reply decode.

- Median complete time: 6.772827042 s.
- Slowest complete time: 6.778486125 s.
- Median time to first token: 0.053242666 s.
- Median Mac decode: 59.380321 tokens/s.
- MCDMA failures: zero.
- Candidate token hash: `27314db9f375132275a8961431c15a1ffd2887b3b1c6b483f7dfb6ffac900416`.
- Local control token hash: `9684efe04b82897eff43afe857da02ad43b3c0b38e0779d81bd1c74a012f9bb6`.
- First different token: 52.
- Result: fail. It missed the 3% speed threshold and changed the answer.

Raw evidence:

- `results/raw/mia-c1-three-machine-20261003T172540Z-95797/summary.json`,
  SHA-256 `77eee41bb79f989e27fd389717f16a8df9d58818a164da8d049f9afae12ea411`.
- `results/raw/mia-c1-three-machine-20261003T172540Z-95797/result.json`,
  SHA-256 `ef50b088a48fc02c049c024a3198f154064447c56f2fc3ee3c62c78b4ca813e0`.

## Proven cause

`LOCALBOUND-001` recreated the original split entirely on the Mac. Normal local
generation and the mirrored handoff boundary first differed at token 24. This
rules out MCDMA and cross-machine transport as the cause: leaving the final
prompt token for the reply decoder changes the greedy state on its own.

- Normal token hash: `9684efe04b82897eff43afe857da02ad43b3c0b38e0779d81bd1c74a012f9bb6`.
- Mirrored-boundary hash: `4b3da8eb92e0efc4bcbc4228900633520a3177c56f3484b283c2800f05a646e6`.
- Raw log: `results/raw/20261003T174801Z-LOCALBOUND-001.log`.
- Raw SHA-256: `2981a280a25054c407e24b31296a7b42403f8e1cf2048c05a1b0c1fae1493467`.

## Corrected boundary proof

`LOCALBOUND-002` cached the complete prompt, computed the first reply token from
the prompt's final hidden row, primed MTP with that reply token, and continued
with the full prompt already cached.

- Exact 400-token match: yes.
- First different token: none.
- Continuation time: 6.706920958 s for 399 remaining tokens.
- Continuation rate: 59.804439 tokens/s.
- First-token preparation: 0.051896541 s.
- Raw log: `results/raw/20261003T175145Z-LOCALBOUND-002.log`.
- Raw SHA-256: `d3820ff7a6a41c094cba50d8d37da4d5f6eeb582a44f7259ea970b4497cfc7b04c`.

Protocol v2 now carries the complete main prompt cache, the prompt-minus-one
MTP cache, the final prompt hidden row, and the Spark reference first token.
The Mac derives that token independently and refuses the handoff if it differs.
The next accepted evidence must come from a fresh live three-machine run.

The first v2 live attempt stopped before measurement because its bundle writer
found the retained v1 cache for the same prompt. Both Spark ranks had computed
the corrected first token, but the old manifest was intentionally incompatible.
The writer now namespaces immutable cache bundles by protocol version, leaving
the v1 evidence untouched. Failure and clean-recovery evidence is preserved at
`results/raw/failed-cache-handoff-cache-handoff-20261003T182914Z-30535/`.

## Current upstream pin

- TensorFold release: 0.6.5.
- Commit: `609ca419abecebdc5a059498a613680bd3aa847f`.
- Git tree: `7b1cee8ad8439f9bdfe2ed0e83c1a3595507ba21`.
- Checked base content tree: `1d7f0f5c3251f65d8bc1d653206d449b97a71a5c7f7480948ef002b1d059e302`.
- Checked patched content tree: `58147488b2f9ba740e9ece1de9fb51d002bfbfeb47f348c5fee3a220b98ee251`.
- MiaAI-Lab v1.5 recipe commit: `1576746a04983b6eded0551dbf22512ee9e95654`.

The GLM family files are unchanged from TensorFold 0.6.4, so the existing
two-file dense-stage patch applies unchanged. No performance improvement is
inferred from the version number; the live benchmark must measure it.

## First protocol-v2 live result

Five deterministic 400-token runs completed at a 6.606843792 s median,
60.415945 decode tokens/s, and 0.002626959 s warm-prefix TTFT. MCDMA reported
zero failures. This was 0.034162335 s slower than the 6.572681457 s acceptance
ceiling. It matched the Mac-only control through token 103 and differed at token
104. The two-Spark same-checkpoint control is now required before classifying
that difference as a correctness failure.

- Evidence: `results/raw/mia-c1-three-machine-20261003T184755Z-43126/`.
- Summary SHA-256: `1ce12ba0c99bc93eb26fb5132aba5dd1a2e1b92870d7db32320732772986955a`.
- Result SHA-256: `3ee7d28adca5afae2f443f2f5a635a15246f56b0c3024fa935cc4280a2d6c720`.

## Same-checkpoint two-Spark correctness control

The portable checkpoint was then run through unmodified TensorFold 0.6.5 on
both Sparks. This uses the model already staged on the cluster; it does not
download or reproduce Mia's separate checkpoint.

- Five deterministic fixed 400-token runs.
- Median complete time: 8.108836417 s.
- Median TTFT: 0.069828292 s.
- Median decode: 49.635103 tokens/s.
- Token SHA-256: `50fe610b08baf4fe5ac942930897d7dd88ffc5e63d9c99d355ead7d5817afdba`.
- Result SHA-256: `f6775fdc466086948bf12ab64920a878ac6b98031e8dc7a9094d212cc8da083a`.
- First difference from the three-machine result: token 90.
- First difference between the three-machine and Mac-only results: token 104.

The three paths are each internally deterministic but do not remain bit-identical
across CUDA prefill, all-CUDA generation, and all-Metal generation. This rules
out an unstable MCDMA transfer; it does not satisfy the current exact-output
acceptance rule. The temporary control was stopped cleanly.

## Zero-yield copy-draft experiment

`MIA-HANDOFF-003` changed one variable from `MIA-HANDOFF-002`: TensorFold's
suffix-copy proposer was disabled on the Mac. In the prior run it made eight
proposals, judged seven, and accepted zero tokens. MTP depth remained one.

- Median complete time: 6.516107958 s.
- Maximum complete time: 6.518934917 s.
- Median TTFT: 0.002737291 s.
- Median decode: 61.258812 tokens/s.
- Required 3%-win ceiling: 6.572681457 s.
- Speed threshold: pass by 0.056573499 s.
- Tail threshold: pass.
- MCDMA failures: zero.
- Stable candidate token SHA-256: `6f1de81f8348f8ddcace1e5071f0f71a21850e96a32cb03915203e50a179a6ed`.
- First difference from Mac-only control: token 104 on every run.
- Prefix setup, excluded from the repeated warm-request timing: 3.506100750 s
  for 148,032,512 bytes, including 2.371220167 s transfer and 1.124623416 s
  import.
- Evidence: `results/raw/mia-c1-three-machine-20261003T191628Z-58465/`.
- Summary SHA-256: `6d950d1bbbd6bcfe0b1b98282d35085aa0f0a3f98d0f32384a61f2278ce87484`.
- Result SHA-256: `044c9c5e157366a3053a38a2cb9107fee305bec503791e0a3529da2022812602`.

This is a verified warm exact-prompt performance win, not final acceptance.
The output rule still fails, and a cold request must count Spark prefill,
MCDMA transfer, and Mac import rather than reusing the prepared prefix.

## DFlash2 cold-handoff experiment — invalidated and retained

Later runtime attestation proved this section's measurements used MTP, not
DFlash2. The deployed official 0.6.5 loader accepted the extra drafter keyword
through `**_` but did not forward it. The preserved startup log says `MTP step`,
and the added telemetry field is null. The raw files and numbers remain below
as retained evidence, but they must not be cited as DFlash2 performance.

TensorFold commit `8debc4eb4df70650c2bf998b7728d039e8e4aaeb` contains the
GLM-compatible DFlash2 decode path intended for this experiment, but that patch
was absent from the deployed Mac source stage. The recipe commit was
`63035efc3f40cb02d485a656c926163883cad264`. No Mia base-model checkpoint was
downloaded: Mia's published result remains the external target. The only new
model artifact on the Studio was the 2,342,169,800-byte DFlash2 helper already
present on the Spark cluster, SHA-256
`b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.

One warmup preceded five measured cold requests. Every measured request rebuilt
the prompt on both Sparks, sent one fresh 148,032,512-byte cache through MCDMA,
and decoded 400 tokens on the Mac with MTP capped at two drafts. The command's
four-bit DFlash2 request was recorded but not activated.

- Median complete time: 7.035067125 s.
- Median time to first token: 0.523886583 s.
- Median Mac decode: 61.279210 tokens/s.
- Median complete-pipeline rate: 56.667062 tokens/s.
- Median MCDMA transfer: 0.518446208 s.
- Steady cache import: 0.004417-0.006111 s.
- Output hash was stable across all five runs.
- MCDMA calls: 35; failures: zero.
- OOM or fatal error signal: none.
- Clean stop: both Spark ranks, the Mac connector, and the MCDMA service were
  confirmed down.

Result: invalidated as a DFlash2 experiment and still a failed goal result.
Decode alone was 1.46% faster than Mia's published 60.4 tokens/s,
but complete time was 3.82% slower than Mia's implied 6.775960265 s and 7.04%
above the required 6.572681457 s ceiling. TTFT was also slower than Mia's
published 0.170 s. The candidate differed deterministically from the Mac-only
control at token 104, consistent with the already documented cross-backend
arithmetic boundary.

Raw evidence:

- `results/raw/mia-c1-dflash-cold-20261003T223636Z-58601/result.json`,
  SHA-256 `bec742211368a9937877bcb80ecef86fed494c7e9089a2b6162b89d24691adb1`.
- `results/raw/mia-c1-dflash-cold-20261003T223636Z-58601/mac.stderr`,
  SHA-256 `574ca45341484d05f36556a6137dc9ba0b4370abeef1e9058cd8be05eecdda63`.
- Clean-stop evidence:
  `results/raw/stopped-cache-handoff-20261003T223933Z-59599/`.

## Positively attested DFlash2 result and compute-ceiling stop

The corrected Mac source stage includes the reviewed GLM DFlash2 patch. Its
full-tree SHA-256 is
`13cf04f5f4975d9f9517f106d4ae993f9fbac0fe1191fb62272c79e80f92b337`,
and the patch SHA-256 is
`76ac8a7256598980e494891a9b3cf332000a3a3a366ed08ef398f87188b8b42e`.
Preflight verified that source and the 2,342,169,800-byte helper on the Mac.
No Mia base-model checkpoint was downloaded.

The five-request cold campaign rebuilt the prompt on both Sparks, transferred
one fresh cache through MCDMA, and decoded 400 tokens on the Mac for every
request. Startup explicitly reported `DFlash2 block 0.0 ms, drafts up to 2`.
Each measured request recorded 170 DFlash proposals and 340 proposed tokens.

- Median complete time: 7.048070291 s.
- Median TTFT: 0.542224500 s.
- Median Mac decode: 61.315624 tokens/s.
- Mia published decode target: 60.4 tokens/s.
- Mia implied complete time: 6.775960265 s.
- Required 3%-win ceiling: 6.572681457 s.
- Miss versus required ceiling: 0.475388834 s.
- MCDMA failures: zero.
- Spark OOM or fatal error signal: none.
- Candidate token SHA-256: `6f1de81f8348f8ddcace1e5071f0f71a21850e96a32cb03915203e50a179a6ed`.
- First difference from the Mac-only control: token 104.
- Raw evidence: `results/raw/mia-c1-three-machine-cold-dflash-20261004T001022Z-91853/`.
- Result SHA-256: `8bcec06ff0ccd27ffa3952e7b889375651ccef667806dac8b45068b342692e73`.
- Summary SHA-256: `986e03a9bdf685b90d4ed92d9e06ba0af8efbfd2e749bd808ab84f29826953ba`.
- Clean-stop evidence: `results/raw/stopped-cache-handoff-20261004T001202Z-92483/`.

The steady handoff was about 0.525 seconds: roughly 0.185 seconds of Spark
prefill, 0.096 seconds for the MCDMA frame call, and the remainder in cache
collection, archive assembly, and verification. The Mac decoder was slightly
faster than Mia's published decoder, but not fast enough to pay that handoff.

A one-load Mac sweep then tested four-bit DFlash depths 1 through 7. Depth two
was fastest at 6.914866 seconds complete and 60.009 decode tokens/s. An
eight-bit sweep of depths 2 through 5 was slower; its best was depth two at
6.998244 seconds and 59.285 decode tokens/s. More accepted draft tokens did not
offset their added work.

- Four-bit sweep evidence:
  `results/raw/mac-dflash-depth-sweep-20261004T001752Z-94749/`.
- Four-bit result SHA-256:
  `813644f65c84560e75971572805dfedecc5a1e86370009b1f2951a26c1c86053`.
- Eight-bit sweep evidence:
  `results/raw/mac-dflash-bits8-sweep-20261004T002224Z-96112/`.
- Eight-bit result SHA-256:
  `5cc56b279b4adcccb9a0f35831027b74a6152db449ee640395d33820425f2e35`.

At the measured 0.542225-second TTFT, the Mac would need 66.164 decode
tokens/s to meet the 3% complete-time ceiling, 7.91% above the measured 61.316.
Even an idealized path retaining only the measured 0.185-second Spark prefill
and 0.096-second wire call requires 63.417 decode tokens/s. This fires the
documented compute-ceiling stop condition: another full-buffer copy reduction
cannot make the frozen short-prompt target pass. The service and all owned
benchmark processes were stopped after collection.
