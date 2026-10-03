# Three-machine concurrent-capacity proof — 2026-10-03

## Result

**PASS.** The Mac Studio and both DGX Sparks served GLM-5.3-Flash at the same
time and beat the same-checkpoint two-Spark service twice from clean starts.
The Mac used official TensorFold `0.6.3`; the Sparks retained Mia's newest
compatible TensorFold CUDA build. MCDMA transferred the real Spark-produced
prompt cache to the Mac.

The fair workload was sixteen simultaneous, identical 256-token requests:

- baseline: all sixteen requests used the fresh two-Spark service;
- candidate: eight used the unchanged two-Spark service while eight decoded
  together on the Mac from one three-frame MCDMA cache transfer; and
- both campaigns used one warmup per arm followed by `ABBABAAB`.

| Clean campaign | Two Sparks | Two Sparks + Mac | Median gain | Conservative lower bound |
| --- | ---: | ---: | ---: | ---: |
| 1 | 137.478 tok/s | 173.095 tok/s | 25.90% | 25.65% |
| 2 | 137.800 tok/s | 172.968 tok/s | 25.71% | 25.23% |

Every one of the eight measured pairs passed the 3% throughput floor. The
candidate's slowest first token was 8.65 seconds or better, compared with
23.51 seconds or better for the corresponding baseline rounds. This is a
frozen shared-prefix capacity result, not a claim about sixteen unrelated
prompts.

## Correctness and machine use

- Every lane returned the frozen 256-token answer exactly.
- Every candidate round used eight Spark lanes and eight Mac lanes.
- Every candidate round made exactly five MCDMA calls: prepare, three cache
  frames, and release. The transferred cache was 148,013,056 bytes.
- Every baseline round made zero MCDMA calls.
- Head, worker, and replay identities, starts, images, settings, health, restart
  counts, and OOM state remained accepted throughout each campaign.
- The two campaigns used new containers and start times but the same frozen
  input, allocation, and effective settings.

The cross-campaign checker initially rejected the clean runs because Docker's
NVIDIA runtime emitted identical unique environment entries in a different
order. Raw Docker inspection proved that no value differed. The checker now
sorts unique environment keys, retains order when a key is duplicated, binds
the raw records to accepted container IDs and start times, and still rejects
any changed value or incomplete evidence. Its canonical output SHA-256 is
`07c56c1532c412848c33574674942f3be58eb4769e66801434f8a2ba0b28ab10`.

## Pinned runtime

- campaign code: `872dce45b5c0b943cfa7c29046a84c3e87b0717c`
- Mac TensorFold: `0.6.3`, upstream commit
  `9356df5c424b0c36b7737e37873a6f968b08de79`
- prepared Mac source tree:
  `72073174bce47f4f2f6e3f7b1f1824ae0e22e4fec46269c421334fa143baf573`
- Spark image ID:
  `sha256:446a23697c7eba9e0434cb162b5389792c7a5ab0f2652173bf1e791e7fdb4c5d`
- portable model snapshot:
  `Vontra/GLM-5.3-Flash-MLX-4bit-MTP@76add2a341a1cd90ad0e86bb69839ea9c35827c6`
- complete model content manifest:
  `b1b218bca5c72e28d77f5f4584b3bf12cacef97ba6df59f45a0094a3a041af07`

## Evidence

Campaign 1:

- raw directory:
  `results/raw/paired-capacity-p16-s8-m8-t256-20261003t072616z-82971/`
- `acceptance.json` SHA-256:
  `0afb89a1432a6ec85d245cf8f3cdf74d67e4561dee693aead94f05e26f4e428a`
- `rounds.jsonl` SHA-256:
  `36844c2099eb6c4b5b430e4e2f0ade2e396a83b9d9ef461dcda017888d65ff4e`

Campaign 2:

- raw directory:
  `results/raw/paired-capacity-p16-s8-m8-t256-20261003t073912z-91464/`
- `acceptance.json` SHA-256:
  `c5843242bc628599b14afa2e89ff53d7874bbc5dc94e589cd9942d74ad965756`
- `rounds.jsonl` SHA-256:
  `5735a0bba7a013a37bf8ffa028d5be8aeb984358b98ad07ab6e07cdeda0d6850`

Both use frozen-reference SHA-256
`3fd4b0774f2c63350511288168d004c9de61904ffda43e07455dd9b6f399a322`.
The final Python suite passed `232/232`. Cleanup left no owned TensorFold,
capacity-worker, cache-replay, or MCDMA process/container running on any of the
three machines.
