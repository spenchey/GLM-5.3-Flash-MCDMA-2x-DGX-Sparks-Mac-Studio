# P06C2 receiver-proof protocol receipt — 2026-10-01

## Outcome

PASS for the offline repair. No live host, installed binary, service, cable, or
network setting was touched.

The bandwidth protocol is now version 3. For READ, the receiving initiator
sends its exact verification result in a strict `COMPLETE` message. The
responder validates that message and reports the same receiver proof. For
WRITE, the responder remains the local verifier and the initiator reports the
proof returned by that peer. After the two final acknowledgements are relayed,
the controller closes both endpoint inputs. Each native endpoint must observe
that EOF before reporting a result; an idle period is never accepted as a
clean close. Any complete or partial trailing record before EOF fails the run.
The controller tracks each endpoint input independently and only declares the
EOF gate complete after both closes return successfully. A close or flush
failure is fatal; exception cleanup retries any still-open input without
replacing the original protocol error.
Every result and CSV row identifies both:

- `verification_source=local|peer`
- `verification_receiver=initiator|responder`

## Exact inputs and identities

- Project base commit: `12749d6b0d9e6249d5c211e85d2f2eadb8773ea5`
- MCDMA base commit: `719219272c9ce6fc091b4eab6214eac510b5e387`
- Cleanup patch SHA-256: `57c5a86174920e392941fe0b1f3209377bb42954896f16c0d721769c57494839`
- Receiver-proof patch SHA-256: `630aeabb8bd9278e0efcf1c090355ce27d1bf09fcfffa0208c50e9b4b013059e`
- Exact-verification manifest SHA-256: `b98ad1c623155ce0754f4b2f51fee82560766a0c8cff0d20e152c2ffeac4e23e`

Key materialized-file SHA-256 values:

- `benchmarks/mcdma_bw.c`: `df903c648fc0db486b88c061a37eceea31c9e85788ed6e73e4ad7597c20e8943`
- `benchmarks/run_bw.py`: `0f27d6cc86b0c1d628a97db3037d009c5944f0422f20ae17de420e03c06dcb5e`
- `benchmarks/verify_windows.h`: `26731a52940927b08f905eb17462ee6c05cb0524f1b988ec31f46d8fe3f587a4`
- `tests/bw_pattern_trial.c`: `a0cc48ad07eb5a4307b7bed5cee6f8d83eeee6797d7d3efe27af19b55e73097c`
- `tests/bw_payload_trial.c`: `5cd216dcef26afd9b78abf6fbde5cb097990125e01558d52e485165960383750`
- `tests/test_bw_pattern_trial.py`: `f363e5aa7e0c4995072bc2a6bdcd8e297c8e08546d63d1bd375de9559cb3ca06`
- `tests/test_bw_payload_trial.py`: `28968b6c3f65a3f7e19b68ef561c82bfbf057366c1f71a4d768fbf515fa490a8`
- `tests/test_bw_tools.py`: `5ea4085ce1fcd6f1bfacd724d0b86301e6d76ea64df2d6f952c986497cc7181c`

## Proof performed

The canonical materializer rejected a wrong base, applied both pinned patches
to a fresh archive of the exact MCDMA base, and matched every file hash in the
manifest.

The canonical exact-verification suite passed **93/93** tests. A second fresh
materialization passed the sanitizer-backed protocol/window suite **25/25**
and the controller plus complete main-to-endpoint-to-relay suite **13/13**
(including the full runner's **7/7** tests).

The production trial branches were exercised offline with a shared-memory
transport and compiler sanitizers. For both READ and WRITE:

- both endpoint result rows reported `verified_bytes=1048576`;
- both endpoint CSV rows reported `verified_bytes=1048576`;
- clean trials reported `mismatches=0` and `guard_ok=1`;
- the actual receiver row reported `verification_source=local`;
- the other endpoint reported `verification_source=peer`;
- forced corruption made both endpoints fail while preserving the exact
  1,048,576-byte receiver-proof count;
- old-version, missing-field, duplicate-field, wrong-receiver,
  contradictory, and wrong-byte-count `COMPLETE` messages failed closed;
- the same malformed-proof cases in `DONE` failed before the initiator could
  report success;
- separately delivered duplicate `COMPLETE` and `DONE` records failed for both
  READ and WRITE before the final acknowledgement;
- at 40, 49, 55, and 80 milliseconds after either final acknowledgement,
  partial trailing writes in both READ and WRITE were either refused by the
  closed controller pipe or accepted then rejected by the native endpoint;
- complete and partial trailing bytes synchronously queued in the same write
  as either `CONFIRM` or `CONFIRMED` made the targeted native endpoint exit 2
  with `protocol_trailing_control` for both READ and WRITE;
- a separate post-close test proved that writes after the controller closed
  each endpoint input were refused;
- duplicate final acknowledgements and arbitrary complete or partial trailing
  records likewise could not produce success;
- both native endpoints timed out and failed when the controller withheld EOF;
- delayed but single valid final acknowledgements followed by clean EOF still
  succeeded;
- the controller test proved both acknowledgements were forwarded before both
  endpoint inputs closed, and endpoint results appeared only after that close;
- injected close and flush failures made the relay fail, per-endpoint retry
  remained idempotent, and a cleanup failure was recorded without masking the
  original strict-close exception;
- an injected controller timeout closed both endpoint inputs before bounded
  endpoint shutdown, avoiding the 15-second wait path;
- fake endpoints that emitted `CONFIRMED` plus a complete or partial extra
  `BW_MSG` in one stdout write failed the complete runner path and produced no
  CSV file;
- forged result/CSV proof, role, direction, and trial rows failed the complete
  runner path, returned nonzero, and produced no CSV file;
- receiver-proof rows with an unplanned warmup flag or extra trial, and a
  `BW_DONE` message claiming two trials when one was planned, failed the
  complete runner path and produced no CSV file;
- a protocol-v2 endpoint was rejected before the data phase.

`git apply --check` passed against a fresh after-cleanup tree before the full
suite ran. The suite also proved that no child process survived its temporary
test scope. Stored blank patch-context lines were normalized to remove trailing
whitespace, and the final branch-wide `git diff --check` is clean.

## Remaining gate

This is offline evidence only. Protocol v3 is intentionally incompatible with
the currently installed protocol-v2 binaries, so both endpoints must be built
from this same manifest and replaced together inside the later controlled P06C
run. P06C must still prove the exact 1,048,576-byte row gate on the real Mac-to-
Spark link before P06C2 can be treated as live-validated.
