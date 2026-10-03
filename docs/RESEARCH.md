# Upstream research and integration plan

> Historical research record. The implementation described in
> `TENSORFOLD-MCDMA-DESIGN.md` now passes on all three machines. Statements
> below about what was not yet implemented describe the 2026-10-01 starting
> point, not the current deployment.

Verified 2026-10-01. Exact research revisions are recorded in
`UPSTREAM.lock`; the production deployment remains on its separate pinned
revisions.

## What each project gives us

### MiaAI GLM TensorFold recipe

Source: <https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold>

Use this as the production baseline. It already owns the two-Spark model,
TensorFold image, RoCE/NCCL split, MTP drafter, one-command launch, and a real
answer check. Preserve its pinned model and image instead of copying its files
into this repository.

### TensorFold

Source: <https://github.com/ashhart/TensorFold>

TensorFold currently selects one compute backend per process. Its GLM CUDA
path communicates between CUDA ranks; it does not provide a mixed Metal/CUDA
stage planner. Adding the Mac Studio is therefore an engine feature, not a
configuration change.

The current standalone release is TensorFold `0.6.3` at commit
`9356df5c424b0c36b7737e37873a6f968b08de79`. The prepared Spark target is Mia
recipe `main` at commit `cf28cc4f8038be322cdeda220c6f1c8ace8f27d1`.
Its published GLM CUDA image remains TensorFold `0.6.0`, now with 68 patches,
digest `sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529`,
and patch label `5e01f1bb74d8`.

The latest standalone engine and latest recipe are therefore not the same
version. The Sparks use Mia's tested patched `0.6.0` pair. The Mac uses official
0.6.3 in its own isolated source stage because that release owns the current
eight-lane MLX engine; the local two-file patch only adds the reviewed EXL3
dense-stage loader. Mia's patches are never applied to 0.6.3.

### MCDMA

Source: <https://github.com/ashhart/MCDMA>

MCDMA supplies the transport libraries, validation tools, `mcdma-rpcd`, and
shared-memory mailboxes. Applications must explicitly use those mailboxes. A
live RDMA link alone does not make TensorFold use Mac memory or compute.

The installed link passed byte-checked validation-client transfers, but neither
that receipt nor upstream's pinned documentation proves `mcdma-rpcd` on the
actual hardware. A daemon echo and fault receipt is therefore an independent
gate before TensorFold integration.

The documented KV handoff supports bf16, fp16, and fp32 and refuses FP8. The
live recipe reports an FP8-KV patch, but the exact patched cache implementation
must be extracted before deciding whether that separate handoff design applies.
The first design does not move KV cache; each layer owner keeps its own cache.

### oMLX

Source: <https://github.com/jundot/omlx>

This is reference code, not the replacement serving engine. It shows the
shortest proven design for moving a Mac/CUDA stage activation over MCDMA:

- PR 3869 is merged, but its own record says that code had not been run end to
  end on ConnectX hardware when merged.
- PR 3870 is open and extends MCDMA to every stage hop and sampled tokens. Its
  later hardware evidence is useful but is not proof for this TensorFold path.
- PR 3941 is open and adds vLLM remote prefill plus KV-cache handoff. It is a
  different application path and does not validate this deployment.

We will reuse the protocol and safety ideas in TensorFold: daemon-owned queue
pairs, byte-checked probes, rank agreement, explicit fallback, and visible
evidence. The deployed API remains TensorFold.

The current TensorFold service is tensor-parallel across two CUDA machines.
Adding the Mac requires a new mixed pipeline layer around that CUDA group. A
future two-Spark composite stage needs an outer Mac/CUDA pipeline plus the
existing inner CUDA communication.

### Inference Engineering

Source: <https://www.baseten.co/inference-engineering/llms.txt>

Philip Kiely's *Inference Engineering* is the local-inference book previously
sent to Jeeves. It directly informs this project:

- establish an unchanged baseline before optimizing;
- measure time to first token, inter-token latency, output tokens per second,
  end-to-end latency, and tail percentiles rather than one headline number;
- keep tensor-parallel work inside the fastest connected group; a per-layer,
  per-token pipeline boundary can lose to communication and idle time;
- use speculative decoding to produce multiple accepted tokens per model pass,
  which is the key mechanism needed to amortize this Mac/Spark boundary;
- change one variable per experiment and repeat runs;
- use load, failure, and canary tests before a serving cutover.

These rules are now enforced by `docs/PROJECT-PLAN.md` and the append-only
experiment log.

### OpenResearch and autoresearch

The optional controller is the Windows desktop, using OpenResearch 0.2.14 and
OpenCode 1.18.34. Its OpenResearch project ID is
`a2f3b60c-cebf-4999-bdba-715a45ee1a41`. The project clone is pinned to
`9e9fa860bdfaadf87963fa20b2cabd4392db8659`.

The controller reached the current GLM endpoint and OpenCode returned exact
`CONTROLLER_READY`. This proves the control path only. It did not change the
model, service, routing, or live configuration. The sanitized proof is the
[OPT1 Windows controller receipt](../results/receipts/OPT1-windows-openresearch-controller-2026-10-01.md).

Karpathy's autoresearch repository was reviewed at commit
`228791fb499afffb54b46200aca536f79142f117`. We reuse its practical experiment
method: freeze the evaluator, change one setting, record every result, and make
an explicit keep or reject decision. We do not use or run its nanochat trainer.
That trainer targets one NVIDIA GPU, while this Windows desktop has no usable
NVIDIA GPU and this project is a distributed inference system.

OpenResearch remains a control and recording layer only. It is not imported by
the model, is not needed by the offline evaluator, and must never automatically
change the live service. Git, Linear, and content-addressed receipts remain the
acceptance record. See `docs/AUTORESEARCH.md` for the offline contract.

## Execution order

1. Extract and hash the live patched TensorFold source, then benchmark the
   unchanged two-Spark service over five fixed seeds.
2. Rehearse the stop/test/restore cycle three times. Two full GLM engines cannot
   safely coexist on the 128 GB Sparks.
3. Prove `mcdma-rpcd` on the real hardware with checksummed payloads and faults.
4. Measure MLX/EXL3 divergence and disk headroom, then choose the checkpoint
   policy instead of assuming a mixed boundary is safe.
5. Freeze the four-stream, Mac-initiated state contract.
6. Build one frame codec, TCP correctness path, and daemon-owned MCDMA path.
7. Prove a full CUDA loopback split through the real codec before Metal work.
8. Prove a three-machine synthetic GLM, then a bounded real ordinary-decode run.
9. Add partial commit, MTP, and joint prefix snapshots as separate changes.
10. Publish sanitized positive and negative results upstream.

## Current hardware facts

- Two Sparks communicate on their separate 200 Gb/s direct link.
- The Mac Studio MCDMA link has passed byte-verified read/write tests.
- Port negotiation was observed at 40 Gb/s, but usable throughput in each
  direction has not been measured on this exact path. Report both facts.
- No result counts as GLM-over-MCDMA until a model-level trace and correct
  answer prove that tensors crossed the MCDMA path.
- The Studio has the complete 43-shard MLX checkpoint locally. Its semantic
  GLM architecture and tokenizer match the Spark checkpoint; numerical
  boundary compatibility is still a required experiment, not an assumption.
