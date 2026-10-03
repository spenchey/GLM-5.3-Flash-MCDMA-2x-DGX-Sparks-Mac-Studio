# P06A Spark bandwidth-tool build receipt

Date: 2026-10-01

Linear issue: `MOT-3360`

Result: **PASS**

The Linux `mcdma-bw` executable was built offline on `<head-spark>` from the
pinned MCDMA source export and installed only in its user-owned install tree.
No daemon or benchmark was started, no data crossed the RDMA link, and no
system or network setting changed.

## Source identity and cleanliness

Source/export root:

```text
/home/<spark-user>/projects/MCDMA-719219272c9c
```

The directory is a source export rather than a Git working tree, so cleanliness
was verified against the authoritative local Git object instead of treating
the absence of `.git` as proof:

- `SOURCE_COMMIT` contained exact commit
  `719219272c9ce6fc091b4eab6214eac510b5e387`.
- The authoritative MCDMA checkout was at that same commit and had an empty
  `git status --short`.
- SHA-256 comparison of every exported tracked file against
  `git show 719219272c9ce6fc091b4eab6214eac510b5e387:<path>` passed for all
  152 files before the build and again after installation.
- A sorted path comparison found no added, missing, or unexpected source file.
  The known `SOURCE_COMMIT`, `.test-bin/`, `build/`, and `install/` deployment
  paths were excluded from the tracked-source comparison.
- No pinned source file was edited.

## Compiler and exact build

Compiler:

```text
cc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
```

Exact build and install commands, run as user `<spark-user>` without `sudo`:

```text
cd /home/<spark-user>/projects/MCDMA-719219272c9c
cc -std=c11 -O2 -Wall -Wextra -Werror \
  benchmarks/mcdma_bw.c -libverbs -o build/mcdma-bw.p06a
install -m 0755 build/mcdma-bw.p06a \
  /home/<spark-user>/projects/MCDMA-719219272c9c/install/bin/mcdma-bw
```

The temporary `build/mcdma-bw.p06a` was removed after the installed artifact
was verified. Nothing was installed under `/usr/local`.

## Installed artifact

- Path:
  `/home/<spark-user>/projects/MCDMA-719219272c9c/install/bin/mcdma-bw`
- Owner: `<spark-user>:<spark-user>`
- Mode: `0755`
- Size: `74,968` bytes
- SHA-256:
  `64d3b2f93022c506faefb12b2134e3bb79894d34fe4589f1d73ed41e34aeb8b9`
- Format: 64-bit ARM aarch64 ELF PIE, dynamically linked, not stripped
- ELF Build ID: `1fca30764ad0f627aabda6d5ee78da72506d78f2`
- Source version: MCDMA commit
  `719219272c9ce6fc091b4eab6214eac510b5e387`; `mcdma-bw` has no separate
  `--version` command.
- Runtime libraries resolved from the Ubuntu system paths:
  `libibverbs.so.1`, `libc.so.6`, `libnl-route-3.so.200`, and
  `libnl-3.so.200`.

The installed binary's offline `--help` parser returned its documented usage
and expected exit code 2 without opening a device:

```text
usage: mcdma-bw --role initiator|responder --device NAME [--gid-index N]
  --op write|read|send --bytes N[K|M] --depth D --qps Q [--cq-per-qp] --total N[K|M|G]
  [--repeats R] [--warmup W] [--mtu 1024|2048|4096] [--finish auto|flag|imm]
  [--payload SOURCE | --dump NEW_DESTINATION] (one resident transfer; see docs)
  [--verify-bytes N] [--max-region N] [--timeout SECONDS] [--psn N]
```

## Offline checks

All selected bandwidth checks ran with bytecode writes disabled and without a
device, peer process, daemon, TCP listener, or network transfer:

| Check | Result |
| --- | --- |
| `tests/test_bw_guard.py` | 1 passed |
| `tests/test_bw_payload.py` | 6 passed |
| `tests/test_bw_payload_trial.py` | 3 passed |
| `tests/test_bw_tools.py` | 13 passed |
| Installed `mcdma-bw --help` | documented usage, expected exit 2 |

Total: **23 offline tests passed**.

## Disk observation

The source and install tree are on `/dev/nvme0n1p2`, mounted at `/`.

| Point | Used KiB | Available KiB | Capacity |
| --- | ---: | ---: | ---: |
| Before | 2,330,775,860 | 1,406,558,812 | 63% |
| After | 2,330,775,960 | 1,406,558,712 | 63% |

The filesystem used-space reading increased by 100 KiB across compilation and
installation. The final installed executable is 74,968 bytes.

## Preservation and final state

Before and after the build:

- head `glm53-flash-tf`: `running=true`, `oom=false`, restart count `0`;
- worker `glm53-flash-tf`: `running=true`, `oom=false`, restart count `0`;
- head `GET /v1/models`: HTTP `200`.

Final cleanup checks found:

- no `mcdma-rpcd` process;
- no `mcdma-bw` process;
- no listener on port 18620;
- no temporary build executable;
- no MCDMA daemon, tunnel, benchmark, network transfer, interface/route/driver
  action, cable action, `sudo`, `/usr/local` write, or GLM stop/restart.

This receipt proves only the pinned offline Spark build. P06C remains
responsible for the separately authorized bounded hardware measurements.
