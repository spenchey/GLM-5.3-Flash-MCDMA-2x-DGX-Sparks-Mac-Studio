# P06 throughput and fault-recovery preflight

Date: 2026-10-01

Linear issue: `MOT-3329`

Result: **PLAN READY; HARDWARE RUN BLOCKED AND NOT STARTED**

This receipt defines the bounded P06 run. It does not claim a throughput or
fault result. No daemon, tunnel, benchmark, fault injection, cable action,
route, interface, firewall, Tailscale setting, driver, GLM process, or MCDMA
source checkout was changed during this preflight.

P05 is the prerequisite proof: the pinned daemons completed exact 32 KiB and
2 MiB request/reply calls on `rdma_mcrdma1` to `rocep1s0f0`, generation stayed
1, both sides recorded two calls and zero failures, control was loopback-only,
and orderly cleanup passed.

## Current blockers before execution

1. The pinned Mac bandwidth executable exists at
   `/Users/<studio-user>/Projects/MCDMA/build/mcdma-bw`, but no executable
   `mcdma-bw` exists under the pinned Spark checkout
   `/home/<spark-user>/projects/MCDMA-719219272c9c`. Build and hash the Spark
   executable from that unchanged checkout before any measurement. Do not use
   a nearby or unpinned binary.
2. The P05 mailbox verifier intentionally supports only two successful calls.
   It has no delayed-service, client-timeout, or prompt-controlled cable-fault
   mode. The actual P06 run needs a separate bounded test helper reviewed to
   ensure it cannot silently continue after a timeout. Do not improvise with
   raw mailbox writes during the hardware session.
3. The cable step requires Spencer at the machines and explicit approval. All
   unattended stages must pass first.

P06 must not begin while any blocker remains.

## Fixed identity and safety gates

Recheck immediately before every stage, not once for the whole session:

- Mac Studio hostname, build `26A428`, `rdma_mcrdma1`, RoCE-v2 GID index 0,
  active interface `mcrdma1`, and pinned provider/checker/binary hashes.
- Spark hostname `<head-spark>`, `rocep1s0f0`, RoCE-v2 GID index 1, active
  interface `enp1s0f0np0`, path MTU 4096, and pinned executable hashes.
- Port 18620, both daemon Unix sockets, daemon mailboxes, and the tunnel state
  must be absent before startup.
- Both `glm53-flash-tf` containers must be running, OOM false, and have the
  same restart counts captured at baseline; `GET /v1/models` must return 200.
- The Spark management path and inter-Spark link must remain distinct from the
  Mac-facing QSFP28 link. Any identity mismatch aborts before starting.

Keep raw output in a new ignored results directory with mode 0700. Record the
OS build, device/GID/interface values, binary SHA-256 values, start/end times,
commands, stdout/stderr, daemon status, helper exit codes, link state, GLM
health, and cleanup evidence. Never publish raw QP keys or memory addresses.

## Stage 1 — both-direction sustained throughput

This stage is unattended after the missing pinned Spark binary is built and
hashed. It uses the repository's `benchmarks/run_bw.py`, not `mcdma-rpcd`.
SSH carries only the benchmark control lines; payloads use the selected RDMA
devices. The exact planned command is:

```text
python3 benchmarks/run_bw.py \
  --mac-host <mac-studio> --peer-host <head-spark> \
  --mac-bw /Users/<studio-user>/Projects/MCDMA/build/mcdma-bw \
  --peer-bw /home/<spark-user>/projects/MCDMA-719219272c9c/install/bin/mcdma-bw \
  --mac-provider /usr/local/lib/rdma/libmcdma-rdmav34.so \
  --mac-checker /Users/<studio-user>/Projects/MCDMA/build/cx5-native-check \
  --mac-interface mcrdma1 --peer-interface enp1s0f0np0 \
  --mac-device rdma_mcrdma1 --peer-device rocep1s0f0 \
  --mac-gid-index 0 --peer-gid-index 1 \
  --ops read,write --initiators mac,peer --sizes 4194304 \
  --depths 1 --qps 1 --cq-modes shared --total 8589934592 \
  --repeats 3 --warmup 1 --mtu 4096 --finish flag \
  --verify-bytes 1048576 --timeout 120 \
  --mac-cq-map 0 --mac-user-post 0 --mac-user-bf 0 \
  --output <new-ignored-results-directory>
```

The preflight `--dry-run` expanded this into exactly 12 benchmark pairs: READ
and WRITE with each endpoint initiating, repeated three times in alternating
order. Each trial is internally limited to 120 seconds.

Required proof for every measured row:

- both endpoint exit codes are zero;
- posting mode is confirmed, no completion error occurred, and the 1 MiB
  verification sample plus guards passed;
- exact payload bytes, elapsed boundary, and Gbit/s are retained in CSV;
- manifest hashes cover all raw logs and CSVs;
- report separate medians/ranges for Mac READ, Mac WRITE, Spark READ, and Spark
  WRITE. Do not convert these payload rates into inference speed claims.

Expected service impact: the dedicated RDMA/Thunderbolt path is saturated in
bounded trials. GLM remains running, but API latency may temporarily rise.
Abort the campaign immediately if either container restarts, reports OOM, the
API stops returning 200, verification fails, or a device/GID changes.

## Stage 2 — bounded RPC timeout without a service

This stage is unattended once the P06 helper exists.

1. Start the Spark daemon on `127.0.0.1:18620`, start the authenticated tunnel,
   then start the Mac connector exactly as P05 did, with direct replies and
   `MCDMA_RPC_PULL` unset.
2. Capture both `STATUS` blocks and Mac mailbox generation. Require `up`,
   `calls 0`, `failures 0`, and generation 1.
3. Do **not** attach a service. The bounded client stages one 32 KiB request
   and waits exactly five seconds for its sequence on the done word.
4. Require a client timeout exit, unchanged generation, no reply file, Mac
   `calls 1 failures 0`, and Spark `calls 0 failures 0 service=none`. These
   counters prove the request was sent but no fabricated reply appeared.
5. Do not attach a service afterward because it could consume the stale
   request. Clean the entire mailbox generation: Mac connector `SHUTDOWN`,
   Spark listener `SHUTDOWN`, tunnel stop, then prove both mailboxes/sockets and
   every test process are gone.

If observed counters differ, preserve logs and stop; do not reinterpret a
different path as a timeout pass.

## Stage 3 — orderly daemon loss and fresh recovery

This stage is unattended and uses a new clean daemon generation.

1. Start listener, tunnel, connector, and service in the P05 order.
2. Complete and byte-check one 32 KiB request/reply. Capture hashes, `calls 1`,
   `failures 0`, and generation 1 on both sides.
3. With no call in flight, send `SHUTDOWN` to the Mac connector. Require the
   Mac log to say every verbs object was destroyed, the Spark log to report
   control closed/link down, and the service to receive `BYE` or EOF.
4. Send `SHUTDOWN` to the Spark listener. Stop the tunnel last. Require every
   process, mailbox, socket, and listener to be absent.
5. Start a wholly fresh listener, tunnel, connector, and service. Complete the
   same byte-checked call and capture new PIDs, QPs, generation 1, matching
   hashes, and zero failures. Shut down in the normal order again.

No `SIGKILL` is permitted. If a Unix control socket is unexpectedly
unreachable, `SIGTERM` is the only fallback because the pinned daemon handles
it through the same orderly teardown path. If it still does not exit within
30 seconds, stop and request assistance; never escalate to a forced kill.

## Stage 4 — control-path loss and same-daemon reconnect

This stage is unattended and intentionally tests reconnect while idle.

1. Start the normal four components and complete one 32 KiB call. Require
   generation 1 and `calls 1 failures 0` on both daemons.
2. Close only the authenticated SSH control tunnel while no request is in
   flight. Do not change routes, interfaces, firewall, Tailscale, or RDMA
   configuration.
3. Require the connector to report link down, the listener to report control
   closed/link down, the registered service to receive `BYE` or EOF, and both
   daemon processes to remain alive. Capture counters before proceeding.
4. Restart the same loopback tunnel. Allow at most 30 seconds for the existing
   connector to reconnect. Require Mac generation 2, both daemons `up`, new QP
   identities, and unchanged failure counters.
5. Attach a fresh service and complete another byte-checked 32 KiB call.
   Require `calls 2 failures 0`, generation 2, and matching hashes.
6. Clean up normally: connector `SHUTDOWN`, listener `SHUTDOWN`, tunnel stop.

Failure to reach generation 2 is a failed reconnect, not permission to restart
processes and call the stage successful.

## Stage 5 — one owner-assisted QSFP28 cable disconnect

Run only after Stages 1–4 pass and Spencer gives the approval quoted below.
The physical target is the **Mac-facing QSFP28 DAC at `<head-spark>` port
`enp1s0f0np0` / `rocep1s0f0`**. It is not the Spark-to-Spark cable and not the
Thunderbolt cable between the Mac Studio and Helios enclosure. If the cable
cannot be identified with certainty, abort without touching it.

1. Recheck GLM health, management reachability, device/GID identity, and the
   exact cable label with Spencer. Start listener, tunnel, connector, and the
   bounded P06 service. Complete one 32 KiB call at generation 1.
2. The client stages a second 32 KiB request with a 30-second deadline. Once
   the runner prints `P06 SAFE TO DISCONNECT MAC-FACING QSFP28 NOW`, Spencer
   unplugs that one QSFP28 DAC from `<head-spark>` for ten seconds and reconnects
   the same plug to the same port when prompted.
3. Expected immediate impact: only the MCDMA data link goes down; management
   SSH, the inter-Spark link, both GLM containers, and the model API remain up.
   The in-flight call must either fail/timeout explicitly or complete with a
   valid full hash; a partial or unverified reply is a failure.
4. Capture timestamped link state, client result, generation, both daemon
   `calls/failures/MiB` counters, service disconnect, completion error/timeout,
   and both logs. Counter outcomes depend on whether the cable is removed
   before the request lands or before the reply lands, so the receipt must
   classify the observed sequence instead of assuming one delta.
5. Allow at most 60 seconds after reconnection for port active/link up and at
   most 30 further seconds for daemon reconnect. Require a generation increase,
   new QPs, then a fresh byte-checked 32 KiB call with matching SHA-256.
6. Stop connector by `SHUTDOWN`, listener by `SHUTDOWN`, and tunnel last.

Abort immediately if the wrong cable moves, management reachability changes,
the inter-Spark link changes, either GLM container restarts/OOMs, the API is not
200, the Mac device disappears, or the link does not recover within the fixed
limits. On abort, reconnect the exact cable if it is out, wait for physical
link state only, then perform the cleanup below. Do not change software or
network settings to force recovery.

## Mandatory abort and cleanup order

Every stage installs a trap before its first process starts. On success,
failure, interrupt, or expired deadline:

1. stop/close the bounded client and service;
2. send `SHUTDOWN` to the Mac connector and wait at most 30 seconds;
3. send `SHUTDOWN` to the Spark listener and wait at most 30 seconds;
4. stop `scripts/mcdma-control-tunnel.sh` last;
5. prove no daemon, helper, benchmark, mailbox, Unix socket, SSH control
   socket, port 18620 listener, or temporary directory remains on either host;
6. recheck both GLM containers, restart counts, OOM state, and API HTTP 200.

If SHUTDOWN is unavailable, use only the daemon's orderly SIGTERM fallback.
Never use SIGKILL, restart GLM, reload the driver, reboot, change a route or
interface, move another cable, or continue to the next stage after a failure.

## Exact approval question for Spencer

> Do you approve one controlled ten-second disconnect of the Mac-facing
> QSFP28 cable plugged into `<head-spark>` port `enp1s0f0np0` / `rocep1s0f0`
> during the bounded P06 test, followed by reconnecting the same plug to the
> same port when prompted? This is not the Spark-to-Spark cable or the
> Thunderbolt enclosure cable. The expected impact is only a brief MCDMA link
> interruption; GLM should stay running. We will abort without touching any
> cable if its identity is uncertain or GLM/management health changes.

Silence or a general request to continue is not approval for the cable step.

## Preflight checks performed

- `benchmarks/run_bw.py --help` and `tools/lifecycle_torture.py --help` parsed.
- The exact Stage 1 command passed `run_bw.py --dry-run`, producing 12 pairs
  without SSH or an output directory.
- The pinned source confirms 10-second daemon handshakes, bounded socket and
  completion waits, generation changes on reconnect, and orderly SHUTDOWN /
  SIGTERM teardown. The lifecycle torture tool was not selected because its
  hang case deliberately uses SIGKILL, which this P06 contract forbids.
- Live read-only checks found no MCDMA daemon, mailbox, tunnel, temporary test
  process, or port 18620 listener after P05 cleanup.
- Both GLM containers remained running, OOM false, restart count 0, and the
  model API returned HTTP 200.

P06 remains blocked until the pinned Spark bandwidth executable and bounded
fault helper exist and Spencer explicitly approves the cable step.
