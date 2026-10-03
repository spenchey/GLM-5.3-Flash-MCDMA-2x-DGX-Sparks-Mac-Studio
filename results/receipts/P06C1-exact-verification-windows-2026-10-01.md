# P06C1 exact verification-window repair

Date: 2026-10-01

Linear issue: `MOT-3366`

Result: **PASS — OFFLINE REPAIR READY; P06C NOT RETRIED**

This work did not start a daemon, open port 18620, run an RDMA transfer, alter
networking, touch a cable, replace an installed binary, or stop/restart GLM.

## Reproducible source

- Exact upstream base: `719219272c9ce6fc091b4eab6214eac510b5e387`.
- Required P06C0 patch SHA-256:
  `57c5a86174920e392941fe0b1f3209377bb42954896f16c0d721769c57494839`.
- P06C1 patch: `patches/mcdma/0002-exact-verification-windows.patch`.
- P06C1 patch SHA-256:
  `69438f5992f256f512eeccc4bf70bad5fd4a2a6f97a33e367c286628d5de91e5`.
- Manifest: `patches/mcdma/exact-verification-manifest.json`, SHA-256
  `2ee798327842100e6f6de2ea5dd813cfaedeafbffa59a09e28c8a936cd77a57a`.
- `scripts/prepare-mcdma-exact-verification.sh` verifies the exact clean base,
  both patch hashes, intermediate P06C0 hashes, final file hashes, and refuses
  an existing destination.

The patched benchmark source SHA-256 is
`3e11e61a3cbdbcbcb74b1d4b9cb77550e76ca1adde9cf410cd04deddf6417aa9`;
the window helper SHA-256 is
`cc18a4f0a45c11815cf77d1023fb8b5dcc670d10e70811e7ce45298acba0e53e`.
The patched runner SHA-256 is
`cecf329dec1adcadac50432ed4ea58f5857266f14c29198dfea5a63f62c4ff23`.

## Contract implemented

- Omitted `--verify-bytes` is reported as auto and resolves to exactly
  `min(1 MiB, agreed resident capacity)`, with the 24-byte minimum enforced.
- Explicit values are never capped: they require at least 24 bytes, eight-byte
  alignment, and no more than agreed capacity, or exit nonzero before traffic.
- Deterministic first/middle/last extents are anchored across the whole
  resident capacity, then split only where a slot boundary requires it. They
  are aligned, unique, non-overlapping and in bounds, sum exactly to the
  effective budget, and use at most three extents per resident slot.
- The exact P06 shape (one QP, depth one, four-MiB request, explicit one-MiB
  verification) produces three extents totaling exactly 1,048,576 bytes.
- Protocol version is now 2. Requested mode/budget is exchanged in ENDPOINT;
  the effective agreed-depth budget is exchanged in VERIFY before QPs enter a
  traffic state. The final `BW_VERIFY agreed=1` line is emitted only after
  depth negotiation and window rebuild. Old, missing, or mismatched values
  fail closed.
- The runner now accepts only the complete v2 ENDPOINT field set. It validates
  protocol, role, operation, bytes, local and planned depth, QPs, CQ mode and
  capacity, total bytes, MTU, immediate-receive mode, atomic depth, PSN, keys,
  addresses, exact resident-plus-guard length, GID/QPNs, payload mode, and
  requested verification mode/budget. Both roles and every plan-consistent
  field must agree before descriptors are forwarded; the exact effective
  VERIFY budget must then agree before any later protocol/data message.
- Successful completion is fail-closed: CSV headers/rows, results, setup,
  finish and `BW_DONE` are rejected before verification negotiation; output
  after `BW_DONE`, duplicate headers/done, CSV-before-header and unknown output
  are rejected. Terminal success requires exactly two validated endpoints, two
  matching VERIFY messages, verified negotiation, and a header, result, row and
  successful `BW_DONE` from each process. A clean child exit without that
  complete state returns nonzero. The deterministic default PSN `0x654321` and
  CQ capacity consistent with granted depth are checked against the runner's
  plan without forbidding legitimate resource clamps.
- Both child streams are drained through EOF. Any leftover unterminated bytes
  fail as truncated, so trailing text or a duplicate `BW_DONE` without a final
  newline cannot disappear. Complete lines are strict ASCII protocol text;
  replacement decoding is forbidden. A valid ASCII record split across pipe
  reads remains accepted, while split non-ASCII bytes return nonzero.
- Resident payload/dump mode uses full CRC64 and constructs no sample windows.
- K/M/G parsing, unsigned narrowing, multiplication, allocation, offset,
  bounds, overlap and exact-sum checks fail closed. Cleanup frees windows.

## Deterministic proof

`tests/test-mcdma-exact-verification.sh` materialized a fresh tree from the
exact base, applied P06C0 then P06C1, verified every manifest hash, rejected a
wrong base, and ran **78 tests; 78 passed**. The suite includes:

- auto/default, zero, minimum, non-multiple, exact one-MiB, capacity boundary,
  over-capacity, arithmetic overflow, allocation failure, determinism,
  remainder distribution, exact sum/alignment/uniqueness/bounds, and
  slot-boundary splitting. The 24-byte two-, three-, and 32-slot fixtures each
  prove offset zero, global midpoint coverage, and final-window end equal to
  resident capacity;
- protocol-version/mode/budget mismatch rejection and runner auto/explicit
  command construction. Real local child processes exercise the complete
  `run_bw.main`/native-endpoint/relay route: a complete matching v2 exchange
  reaches result output, while v1, a missing v2 field, endpoint mismatch,
  effective-budget mismatch, and a valid pair inconsistent with the planned
  command all return nonzero before a data/result row. Additional full-main
  fixtures prove premature CSV/result/done, clean incomplete exit,
  CSV-before-header, duplicate done, and matching alternate PSNs all return
  nonzero and create no CSV. The same full-main path rejects trailing text and
  duplicate done without a newline, plus split non-ASCII bytes, while accepting
  a valid ASCII header split across writes;
- a sanitizer-built shared-memory verbs double that invokes the actual
  `run_initiator_trial()` READ-receiver and `run_responder_trial()`
  WRITE-receiver state machines. Both good paths pass and sampled receiver
  corruption makes both branches return nonzero;
- full-CRC payload bypass and all prior focused/P06C0 cleanup suites.

Materialization log SHA-256:
`04b14245d52c9755f9daadbe1b88d8387e5a4120a64370bc4ab846942dbf5fb4`.
No test child survived.

## Isolated platform builds

Only new mode-0700 `/tmp` trees were used; authoritative binaries were not
overwritten. Both compilers used `-std=c11 -O2 -Wall -Wextra -Werror`.

- Mac Studio build: source
  `3e11e61a3cbdbcbcb74b1d4b9cb77550e76ca1adde9cf410cd04deddf6417aa9`,
  header `cc18a4f0a45c11815cf77d1023fb8b5dcc670d10e70811e7ce45298acba0e53e`,
  binary `8b07c05698a5ea9fd7b37d7dc2df0c55a42b7d121018262185ef7c5174062081`.
- Spark build: same source/header, binary
  `7f3ae470ee5d011beacdb7bcd390529b087784731147b3287d7c0c6d4bebe524`.

On both builds explicit values `0`, `15`, unaligned `1,048,577`, and
over-capacity `4,194,312` returned exact exit code 2 before device open. Each
temporary build tree was removed and no build process remained.

## Installed state and service health

- Installed Mac bandwidth binary stayed
  `de919c4906b62cb29a847c1a26b8fe965f054569edfe6f20ff0b9b97a5057d0f`.
- Installed Spark bandwidth binary stayed
  `64d3b2f93022c506faefb12b2134e3bb79894d34fe4589f1d73ed41e34aeb8b9`.
- No `mcdma-bw`, `mcdma-rpcd`, or port-18620 listener remained on Mac or head.
- Head and worker `glm53-flash-tf`: running `true`, OOM `false`, restart count
  `0`; head `/v1/models`: HTTP `200`.

Final health/cleanup log SHA-256:
`a49877a4ba9140bfa4933733af4dee196980635abdd411617667d0395c968afc`.

## Remaining risk

This task deliberately proved only reproducible source, isolated builds and
offline protocol behavior. The installed endpoint binaries remain the old
hashes above, so P06C cannot resume until a separately authorized runner stages
both isolated v2 binaries, rechecks those exact hashes, and repeats the live
identity/health gates. No throughput or hardware behavior is inferred from the
offline pass.

P06C may now be reviewed for a separately authorized retry; this task did not
perform that retry.
