# P04A Linux `mcdma-rpcd` build receipt

Date: 2026-10-01

Result: **PASS**

The pinned MCDMA Linux daemon was built and offline-tested on `<head-spark>`
without starting it or changing the live host network. The build source was an
archive of exact commit `719219272c9ce6fc091b4eab6214eac510b5e387`, matching
`UPSTREAM.lock` and the current MCDMA checkout recorded by P04.

## Installed artifact

- User-owned source/build root:
  `/home/<spark-user>/projects/MCDMA-719219272c9c`
- Installed binary:
  `/home/<spark-user>/projects/MCDMA-719219272c9c/install/bin/mcdma-rpcd`
- Owner and mode: `<spark-user>:<spark-user>`, `0755`
- File: 64-bit ARM aarch64 ELF PIE, dynamically linked, 76,608 bytes
- Version: `mcdma-rpcd 1.0.0 protocol 1`
- SHA-256:
  `dd56f9f58242dc825cf8e9fefcbb4e99640afb76cc729d2a55ece0f4808acc1a`

The helper library and header were installed beside it under the same
user-owned `install/lib` and `install/include` tree. Nothing was written to
`/usr/local`, no privileged command was used, and no daemon was started.

## Exact build and offline tests

From the pinned remote source root:

```text
PATH=/home/<spark-user>/projects/MCDMA-719219272c9c/.test-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin make -C rpc test
PATH=/home/<spark-user>/projects/MCDMA-719219272c9c/.test-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin make -C rpc all
PATH=/home/<spark-user>/projects/MCDMA-719219272c9c/.test-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin make -C rpc install PREFIX=/home/<spark-user>/projects/MCDMA-719219272c9c/install
```

Results:

- `test_rpc_helper: ok`
- `test_rpcd_socket: ok`
- `tests/test_rpcd_daemon.py`: 10 tests passed in 15.589 seconds
- daemon and `libmcdma-rpc.so` built successfully with `-Wall -Wextra -Werror`
- installed binary returned the expected version and hash above

Ubuntu GCC 13's fortified `snprintf` analysis initially emitted a false
`format-truncation` diagnostic for the validated, maximum-20-character peer
name and `-Werror` stopped the daemon test build. The committed test-only
compiler shim `scripts/tests/mcdma-linux-cc` keeps every upstream warning and
suppresses only `-Wformat-truncation`. No pinned MCDMA source was changed.

## Live service preservation

Before the build on the head Spark:

- `glm53-flash-tf`: `running=true`, `oom=false`, restart count `0`
- `GET http://127.0.0.1:8888/v1/models`: HTTP `200`

After the build and cleanup:

- head `glm53-flash-tf`: `running=true`, `oom=false`, restart count `0`
- worker `glm53-flash-tf`: `running=true`, `oom=false`, restart count `0`
- head `GET /v1/models`: HTTP `200`
- no `mcdma-rpcd` or offline `rpcd-stub` process remained
- no user-owned `rpcd-*` test directory/socket or `mcdma-rpc.*` test file
  remained in `/tmp`

The work did not stop, restart, or modify either live container. It did not
change an interface, address, route, firewall, daemon configuration, or system
installation path. This receipt proves only the pinned Linux build and offline
tests; it does not claim a running daemon or live MCDMA transfer.
