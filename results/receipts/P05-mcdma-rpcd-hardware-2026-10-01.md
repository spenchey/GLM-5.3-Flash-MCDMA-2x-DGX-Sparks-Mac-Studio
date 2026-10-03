# P05 real `mcdma-rpcd` hardware receipt

Date: 2026-10-01

Linear issue: `MOT-3330`

Result: **PASS**

The pinned `mcdma-rpcd` binaries moved deterministic 32 KiB and 2 MiB
requests and equally sized replies through their actual daemon-owned shared
memory mailboxes over the live Mac Studio to `<head-spark>` MCDMA link. Every
byte and every SHA-256 digest matched.

## Fixed endpoints rechecked before startup

- Mac Studio: `spencers-Mac-Studio.local`, macOS build `26A428`, device
  `rdma_mcrdma1`, active RoCE-v2 GID index `0`, interface `mcrdma1` up with
  MTU 9000.
- Head Spark: `<head-spark>`, device `rocep1s0f0`, active port 1, RoCE-v2 GID
  index `1`, active path MTU 4096, interface `enp1s0f0np0` up.
- Mac daemon: `/Users/<studio-user>/Projects/MCDMA/build/mcdma-rpcd`,
  `mcdma-rpcd 1.0.0 protocol 1`, SHA-256
  `950c7cace2b30c8d45070d61938f4f7b6d80496ab961b4dbf4ed208e4645658b`.
- Spark daemon:
  `/home/<spark-user>/projects/MCDMA-719219272c9c/install/bin/mcdma-rpcd`,
  `mcdma-rpcd 1.0.0 protocol 1`, SHA-256
  `dd56f9f58242dc825cf8e9fefcbb4e99640afb76cc729d2a55ece0f4808acc1a`.

No daemon, mailbox, Unix socket, port 18620 listener, or tunnel was present
before startup.

## Exact bounded run

The checked-in `scripts/tests/mcdma-rpc-mailbox.c` verifier was compiled with:

```text
cc -std=c11 -O2 -Wall -Wextra -Werror scripts/tests/mcdma-rpc-mailbox.c
```

The session used this order:

```text
# Spark: control is loopback-only; RDMA uses the observed device and GID.
mcdma-rpcd listen p05 rocep1s0f0 1 4096 127.0.0.1:18620 4 4

# Mac Studio: the checked-in helper creates an authenticated loopback forward.
MCDMA_CONTROL_STATE_DIR=<owner-only-session-state> scripts/mcdma-control-tunnel.sh start

# Mac Studio: MCDMA_RPC_PULL was explicitly unset, preventing pull fallback.
env -u MCDMA_RPC_PULL mcdma-rpcd connect p05,127.0.0.1,18620,rdma_mcrdma1,0,4096,4,4

# Spark service, then Mac client; both are capped at 90 seconds and exactly two calls.
mcdma-rpc-mailbox service p05 /tmp/mcdma-rpcd.p05.sock <artifacts> 90
mcdma-rpc-mailbox client  p05 /tmp/mcdma-rpcd.sock     <artifacts> 90
```

The verifier opens `/mcdma-rpc.p05` with `shm_open`, reads the daemon-published
mailbox layout, registers the Spark side with `MODE poll`, uses the protocol-1
request, staged-reply, and done words, and checks every byte against an
independently reproducible pattern. It permits only the two required payload
sizes and exits after two calls or its bounded timeout.

## Byte and digest proof

| Transfer | Bytes | Mac SHA-256 | Spark SHA-256 | Result |
| --- | ---: | --- | --- | --- |
| Request | 32,768 | `66c60468130441e8b45cdf0e7e79ae9e74c08bf3c84d05e4d40ded131e363b9c` | same | byte-for-byte pass |
| Reply | 32,768 | `cf7ef6b9110e2389d0140a03d32bf910de7a02b6ea8f39283cc237eed1776096` | same | byte-for-byte pass |
| Request | 2,097,152 | `bb25014d9187ad34f594906365c36578067d0ffcabe23df685c4a747358e8654` | same | byte-for-byte pass |
| Reply | 2,097,152 | `e638930f7372624928b00267749019fd8af08160a965457558bf9c2ba3ed18e0` | same | byte-for-byte pass |

Client observations:

```text
CLIENT_PASS size=32768 seq=1 generation=1
CLIENT_PASS size=2097152 seq=2 generation=1
```

Service observations:

```text
SERVICE_PASS size=32768 seq=1
SERVICE_PASS size=2097152 seq=2
```

Both daemon status counters moved from `calls 0 failures 0 MiB 0` to
`calls 2 failures 0 MiB 2`. `MiB` is an integer counter, so it reports the
floor of the exact 2,129,920 payload bytes handled in each direction. The
helper's exact sizes and SHA-256 evidence above preserve the non-rounded
proof. The Mac mailbox generation stayed `1` for both calls.

## Actual RDMA path and no fallback

The Mac daemon logged:

```text
MCDMA_CQ_OBSERVER mapped=1 bytes=16384 readonly=1
mcdma-rpcd: p05: connected (rdma_mcrdma1 qpn 70 <-> 396, mailbox 4 + 4 MiB, direct replies)
```

The Spark daemon independently logged:

```text
mcdma-rpcd: listen p05: rocep1s0f0 gid 1 mtu 4096, control 127.0.0.1:18620
mcdma-rpcd: listen p05: peer ready (direct replies, 1 reply segments)
mcdma-rpcd: listen p05: control connection closed after 2 calls
```

Thus the TCP/SSH path carried only the handshake and control stream. The
daemon QPs used the rechecked `rdma_mcrdma1` and `rocep1s0f0` devices, and the
reply mode was direct rather than the optional `MCDMA_RPC_PULL=1` fallback.

## Isolation, preservation, and cleanup

During the run, listener inspection showed only:

```text
Mac Studio: TCP 127.0.0.1:18620 (LISTEN), owned by ssh
Head Spark: 127.0.0.1:18620
```

No LAN, tailnet, wildcard, or IPv6 listener existed on port 18620. No route,
address, interface, firewall, Tailscale, driver, cable, security setting,
MCDMA source checkout, or GLM service was changed.

Shutdown used the required order and no signal:

1. Mac connector received `SHUTDOWN` and logged `every verbs object destroyed`.
2. Spark listener received `SHUTDOWN` and logged `every verbs object destroyed`.
3. `scripts/mcdma-control-tunnel.sh stop` returned `cleanup=complete`.

The final live check found no daemon process, service process, daemon mailbox,
daemon Unix socket, port 18620 listener, SSH control socket, or tunnel on either
host. Both `glm53-flash-tf` containers remained `running=true`, `oom=false`,
restart count `0`, and `GET /v1/models` returned HTTP `200` after cleanup.

## Limitation

This receipt proves the P05 request/reply transport contract on one real link.
It does not claim throughput, fault recovery, cable-fault behavior, two-link
operation, or TensorFold integration; those remain separate later phases.
