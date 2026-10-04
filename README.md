# GLM-5.3 Flash across a Mac Studio and two DGX Sparks

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

This is an independent integration recipe. It is not an official release of
Mia's AI Lab, TensorFold, MCDMA, oMLX, NVIDIA, Apple, or Z.ai. It exists because
those projects made the underlying model, engines, transport, and research
methods available. See [CREDITS.md](CREDITS.md) and [NOTICE](NOTICE) before
redistributing it.

This repository runs one GLM-5.3 Flash request across all three machines:

```text
OpenAI-compatible request
        |
        v
Mac Studio: tokenizer + TensorFold Metal embedding and original layer 0
        |
        | MCDMA: checksum-protected BF16 [rows, 16384] activations
        v
Spark head: TensorFold CUDA original layers 1..44
        <==== TensorFold NCCL over the private 200 Gb/s link ====>
Spark worker: TensorFold CUDA original layers 1..44
        |
        v
Spark head: final norm + vocabulary head + greedy token selection
```

The Mac and both Sparks keep their own model state. A separate acknowledgement
commits each step on all three machines only after the token returns to the
Mac. Corrupt, stale, duplicate, oversized, timed-out, and out-of-order frames
are rejected instead of silently falling back to another transport.

## Verified status

The original three-machine proof service previously ran and passed the
following gates. It is kept stopped between experiments; a stopped service is
not described as currently live:

- both Spark ranks use the same pinned TensorFold 0.6.0 image and Mia patch set;
- the Mac runs the real TensorFold Metal embedding and original GLM layer 0;
- both Sparks run original GLM layers 1 through 44 through TensorFold CUDA;
- every model activation crosses the real MCDMA hardware link;
- a 21-position answer produced ` 1, 2, 3, 4, 5. ` exactly;
- the persistent API returned exactly `THREE_MACHINE_READY`;
- three clean stop/start/answer cycles passed;
- its recorded link counters exceeded 1,500 calls with zero transfer failures;
  and
- a five-run, 128-token benchmark produced identical output every time at a
  median 30.89 tokens/second in the retained configuration.

The original proof-first service intentionally supports one non-streaming greedy
request at a time. It proved the hardware boundary but is not the performance
target. A separate capacity path keeps the latest compatible Mia CUDA server
on both Sparks while official TensorFold 0.6.3 served eight shared decode
streams on the Mac from one MCDMA-transferred prompt cache. Across two clean
sixteen-request campaigns it reached 172.97-173.09 aggregate tokens/s versus
137.48-137.80 for the same-checkpoint two-Spark service, a 25.7-25.9% median
paired gain with exact output in every lane. That result does not prove the
active goal: one request prefilling on both Sparks, moving its real cache over
MCDMA, and decoding on the Mac faster than the two-Spark service alone. The
earlier cache-handoff form was slower, and that single-request goal remains
open under [its own scorecard](docs/GOAL-SINGLE-STREAM-MCDMA.md).
Temperature sampling, prefix reuse, vision, tool calling, and the full Mia API
surface are not claimed by this first verified implementation. Prompt plus
requested output must fit the verified 2,051-position dense context; sparse
long-context state is not implemented in the split CUDA engine yet.

## Pinned compatible versions

- Published two-Spark target: Mia recipe v1.5, commit
  `1576746a04983b6eded0551dbf22512ee9e95654`; its pinned image uses
  TensorFold 0.6.0 with 70 recipe patches. Its published C1 one-request
  measurements are the external performance target; this project does not
  need another 176 GB copy of Mia's checkpoint merely to restate those claims.
- Spark TensorFold: `0.6.0`, Mia v1.5 patch label `9f73cca659a1`. A controlled
  five-request comparison kept this current image for upstream currency, but
  it improved the one-request median by only about 5 ms and is not presented
  as a performance win.
- Mac TensorFold target: official `0.6.5`, commit
  `609ca419abecebdc5a059498a613680bd3aa847f`, plus the checked two-file EXL3
  dense-stage reader patch. The retained eight-stream receipt was produced on
  0.6.3 and is not silently relabeled as a 0.6.5 result.
- Original layer-split EXL3 model snapshot:
  `9eaebb7c4e96d983dcd538e18624622ba5b820a8`.
- Exact portable capacity checkpoint on all three hosts:
  `Vontra/GLM-5.3-Flash-MLX-4bit-MTP@76add2a341a1cd90ad0e86bb69839ea9c35827c6`.
- Live MCDMA source: `e672c14ff9fc7b38994caf73025cf1588b4de74e`;
  the older `7192192` tree remains only as the exact base for retained offline
  verification-patch tests.

The Mac and Spark package versions intentionally differ. Mia's tested CUDA
recipe remains TensorFold 0.6.0 plus its own 70 patches. The Mac target is the
current official 0.6.5 Metal engine in an isolated, hash-pinned stage; Mia's
CUDA patches are never applied to it. A result keeps the engine version it was
actually measured on, so the older 0.6.3 capacity receipt remains historical.

## Operate it

Copy `config.example.env` to the ignored `config.local.env` and insert the
private host paths and verified hashes. The scripts never download or silently
replace a pinned model or image.

```bash
./scripts/stop-three-machine.sh  # only when an older copy is already running
./scripts/deploy-three-machine.sh
./scripts/preflight-three-machine.sh
./scripts/start-three-machine.sh
./scripts/status-three-machine.sh
./scripts/chat-three-machine.sh "Reply with exactly THREE_MACHINE_READY" 64
./scripts/benchmark-three-machine.sh --repeats 5 --warmups 1 --max-tokens 128
./scripts/stop-three-machine.sh
# Only when a rank, API, or transport component has failed:
./scripts/recover-three-machine.sh
```

`start-three-machine.sh` is idempotent when the verified service is already
running. `stop-three-machine.sh` first stops the Mac API, then uses MCDMA to
shut down both ranks, and finally removes the owned MCDMA processes and tunnel.
`recover-three-machine.sh` refuses a healthy service; after a partial failure it
preserves logs, verifies exact process/container ownership, cleans the failed
deployment, restarts all three machines, and requires an exact model answer.

## Evidence and implementation

- [Current architecture](docs/TENSORFOLD-MCDMA-DESIGN.md)
- [Active single-request performance goal](docs/GOAL-SINGLE-STREAM-MCDMA.md)
- [MiaAI and tonyd2wild control research](docs/RESEARCH-MIA-TONY-SINGLE-STREAM.md)
- [Operator runbook](docs/THREE-MACHINE-RUNBOOK.md)
- [Implementation map](docs/IMPLEMENTATION-MAP.md)
- [Append-only decisions](docs/DECISIONS.md)
- [Append-only experiment log](docs/EXPERIMENT-LOG.md)
- [Inference proof](results/receipts/2026-10-02-three-machine-inference.md)
- [Recovery proof](results/receipts/2026-10-02-three-machine-recovery.md)
- [Performance proof](results/receipts/2026-10-02-three-machine-performance.md)
- [Mac TensorFold 0.6.3 eight-stream proof](results/receipts/2026-10-02-tensorfold-063-mac-eight-stream.md)
- [Three-machine concurrent-capacity proof](results/receipts/2026-10-03-three-machine-concurrent-capacity.md)

Raw logs stay under ignored `results/raw/`. Receipts contain their SHA-256
hashes. Private hostnames, addresses, usernames, credentials, and home paths do
not belong in committed documentation.

## Upstream scope

The reusable parts are the fixed frame protocol, MCDMA mailbox, transactional
cache commit, Metal stage loader, partial CUDA runner, lifecycle scripts, and
tests. Private machine configuration remains local. DFlash2 is allowed only in
this proof-of-concept baseline and is recorded explicitly; any production
release still needs a separate license or replacement decision.

## Credits and license

The recipe's own code and documentation are Apache-2.0. Patch context and
third-party components retain their upstream licenses. Model weights and
container images are downloaded separately and are not part of this repository.

- [CREDITS.md](CREDITS.md) names the projects and people this work builds on.
- [NOTICE](NOTICE) preserves the required third-party notices.
- [Upstream contribution plan](docs/UPSTREAM-CONTRIBUTIONS.md) separates the
  publishable recipe from small changes suitable for TensorFold or MCDMA.

If a credit is missing or wrong, please open an issue before copying the
material further.
