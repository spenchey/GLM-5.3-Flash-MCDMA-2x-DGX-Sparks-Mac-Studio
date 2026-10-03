# P01 unchanged two-Spark baseline receipt

Date: 2026-10-01

Linear issue: `MOT-3326`

Result: **PASS**

The unchanged live TensorFold service completed three fixed prompts at each of
five fixed seeds (`1701` through `1705`). All 15 answers were correct and ended
normally. No container, route, configuration, image, or model file was changed.

## Method

The benchmark used `scripts/benchmark-baseline.sh`, a temporary SSH tunnel to
the existing head-Spark API, and OpenAI-compatible streaming requests with
temperature zero and a 128-token limit. The exact prompt set is retained in
`metrics.json` in the ignored evidence bundle. Its expected visible answers
were `BASELINE_FIXED_OK`, `56`, and `TENSOR FOLD BASELINE`; every seed returned
those answers exactly.

TTFT is request start to the first non-empty streamed model delta. Effective
inter-token latency is decode duration divided by all output-token intervals;
output rate is its inverse. End-to-end time is request start through the final
stream event. Because these Sparks use unified memory and `nvidia-smi` reports
GPU memory as `N/A`, memory high-water marks are the measured whole-host used
memory (`MemTotal - MemAvailable`) during each run, with the nearest surrounding
quarter-second sample included for sub-second runs. They therefore include the
model plus operating-system use and must not be described as dedicated VRAM.

## Per-run proof

All times are milliseconds except output rate, which is tokens per second, and
memory, which is GiB. `API` was HTTP 200 immediately before and after every run.

| Run | Raw SSE SHA-256 | TTFT | Token latency | Output rate | End-to-end | Head HWM | Worker HWM | Page-migration class |
|---|---|---:|---:|---:|---:|---:|---:|---|
| exact-seed-1701 | `21acd2471ed07bdb4eafb33fa1fbdd28e009caa1d2dc5f723e47dc38e50b894c` | 334 | 14.5 | 69.1 | 1165 | 96.9 | 101.7 | normal |
| reason-seed-1701 | `a270def0ae637d6159a553ee86eedd7c445879d8d22c88d5e9333332ca74acba` | 385 | 7.8 | 128.2 | 545 | 96.9 | 101.7 | normal |
| transform-seed-1701 | `1f8d5f5012ac39524a885e4da012e105a32e8e60709c966fa2cb356de09baa91` | 233 | 19.0 | 52.7 | 1072 | 96.9 | 101.7 | normal |
| exact-seed-1702 | `84318a61dfb396cfb698c116f69ff5987921dc6dd20b51e0bc72a7576b9dd5d6` | 210 | 16.7 | 59.8 | 1113 | 96.9 | 101.7 | normal |
| reason-seed-1702 | `cd10da057d8d215fe9b02236b852739883816ebebb47c9d9e9dc946771100cd4` | 326 | 15.0 | 66.7 | 606 | 96.9 | 101.7 | normal |
| transform-seed-1702 | `3999af3ab699a1fcccfe8d1b368745d0435abe9f9d0dbc6f0f576de67a682a47` | 311 | 14.9 | 67.0 | 958 | 96.9 | 101.7 | normal |
| exact-seed-1703 | `d3116ebbc9bbf758d03361da552835eab20da50871d5af86b4548d9567fe54c0` | 261 | 17.1 | 58.5 | 1174 | 96.9 | 101.7 | normal |
| reason-seed-1703 | `0764fce93f6272fa359f533b78d54783a55beacc61c6928e000f1c1c6714dcac` | 282 | 15.1 | 66.2 | 560 | 96.9 | 101.7 | normal |
| transform-seed-1703 | `189094f9bb49a555a6be2c8b52b4f157d4014cdd56e4928df0dd577d828d6074` | 320 | 14.5 | 68.9 | 929 | 96.9 | 101.7 | normal |
| exact-seed-1704 | `b4c9e58dc0580c0a08963fa5b0f035754d6ef8497ab3fa367999cfb72e207b7a` | 514 | 19.8 | 50.6 | 1600 | 96.9 | 101.7 | normal |
| reason-seed-1704 | `9f5dca704fa55870febdeaf54ea17edf0dfa8ab4c0bac95286b4fc67656c56c6` | 414 | 8.7 | 115.3 | 652 | 96.9 | 101.7 | normal |
| transform-seed-1704 | `4d1fdfecfa2ac1cbbbde925b6455af2dc9f35e9a16542075ded0632777eb4d28` | 316 | 14.6 | 68.6 | 928 | 96.9 | 101.7 | normal |
| exact-seed-1705 | `c9906798176c41c08850f067fc9a880fcb992b2255692505007c7c93ecc48ed9` | 260 | 17.4 | 57.4 | 1188 | 96.9 | 101.7 | normal |
| reason-seed-1705 | `eae24b60caecc23f900b0c3ddd40fa7dbd04f6936b8c81787f1ec1f0f1ff9b59` | 414 | 21.8 | 46.0 | 784 | 96.9 | 101.7 | normal |
| transform-seed-1705 | `9ca523ff19761ade3cf41568abf3eb8ae27a911ab6c0e5e04ea2092a6e9fd741` | 253 | 15.8 | 63.3 | 916 | 96.9 | 101.7 | normal |

The cross-run medians were approximately 316 ms TTFT, 15.1 ms effective token
latency, 66.2 output tokens/s, and 0.929 seconds end-to-end.

## Page-migration classification

Classification is per prompt so prompt complexity is not mistaken for memory
movement. A run is labelled `suspected-page-migration-slow` when TTFT exceeds
the greater of twice that prompt's median, its median plus six median absolute
deviations, or 2.0 seconds. All 15 runs were below their 2.0-second thresholds,
so all are explicitly classified `normal`; none was silently folded into an
average. This is a conservative symptom label, not a claim that Linux exposes a
direct page-migration counter for this unified-memory workload.

## Health and retained evidence

Before and after the complete benchmark:

- head container: `running=true`, `oom=false`, restart count `0`;
- worker container: `running=true`, `oom=false`, restart count `0`;
- `/v1/models`: HTTP 200 and `GLM-5.3-Flash-EXL3`, owned by TensorFold.

Every run also recorded HTTP 200 immediately before and after its generation,
and every generation reported `finish_reason=stop`. After capture,
`./scripts/status.sh` still reported both containers running with zero restarts,
the correct API model, and the Studio interface up; `./tests/test-config.sh`
printed `PASS: pinned inputs and private-data guard`.

Canonical ignored evidence:
`results/raw/P01-20261001T171910Z/`

- `metrics.json` SHA-256:
  `3d09088638a605838dbf49703f0405516aaa53ccf6abbee9d47bf2a05e211b1b`
- evidence manifest SHA-256:
  `9c7d6277e61f3809bf3fa032e2f49b5c8e902bcd291c93b69013952531235644`

The temporary tunnel and memory samplers were closed after capture. P01's pass
proof is fully satisfied.
