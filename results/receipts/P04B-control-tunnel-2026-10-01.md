# P04B authenticated loopback control tunnel receipt

Date: 2026-10-01

Linear issue: `MOT-3355`

Result: **PASS**

The fixed MCDMA control tunnel passed an isolated fixture test without starting
`mcdma-rpcd`, changing host networking, or touching either GLM container.

## Fixed design

- Spark control listener: `127.0.0.1:18620`
- Mac forward listener: `127.0.0.1:18620`
- Authenticated SSH target: `<head-spark>`
- Required SSH gates: `ExitOnForwardFailure=yes`, `GatewayPorts=no`, batch
  authentication, keepalive with bounded failure detection
- Local state: project `.state/mcdma-control-tunnel`, mode `0700`, owned by the
  invoking Mac user; the SSH control socket is owner-only
- Startup order: Spark loopback listener, SSH forward, then the future Mac
  connector
- Orderly shutdown: future Mac connector `SHUTDOWN`, Spark daemon `SHUTDOWN`,
  then `scripts/mcdma-control-tunnel.sh stop`

The helper supports `status`, `dry-run`, `start`, and `stop`. It refuses a
pre-existing Mac listener, a pre-existing control path, a missing Spark
listener, or any Spark `18620` listener exposed beyond loopback.

## Isolated proof

Command:

```text
./tests/test-mcdma-control-tunnel.sh
```

Observed result:

```text
PASS: authenticated loopback tunnel round-trip, isolation, fail-closed stop, conflicts, and cleanup
```

The test used a temporary, unprivileged HTTP fixture bound to Spark
`127.0.0.1:18620`. A request through Mac `127.0.0.1:18620` returned the exact
fixture marker. Listener inspection on both machines found only loopback, and
connection attempts to every observed non-loopback Mac IPv4 address failed.
Stopping through the SSH control socket immediately made the Mac endpoint
unreachable. The fixture process/directory, SSH process, Mac listener, state
directory, and control socket were all absent after cleanup.

Conflict tests separately proved that an occupied Mac port and an existing
control path fail closed instead of being replaced.

## Preservation and final state

After the proof:

- both `glm53-flash-tf` containers were still `running=true`, `oom=false`,
  restart count `0`;
- the head `GET /v1/models` returned HTTP `200`;
- no process listened on port `18620` on the Mac or head Spark;
- no fixture, SSH control process, state directory, or control socket remained.

No address, route, firewall, Tailscale setting, driver, cable, RDMA interface,
or live container was changed. No MCDMA daemon was started.
