# P04B MCDMA control-endpoint preflight

Date: 2026-10-01

Result: **DESIGN READY; NOT YET TESTED**

No host, route, interface, firewall, daemon, container, or service was changed.
This receipt records the read-only decision that P04B must test before P04 can
be rerun.

## Decision

Use an authenticated SSH local forward for the daemon's small TCP control
connection. Keep all activation data on the direct RDMA link.

- Spark listener control bind: `127.0.0.1:18620`
- Mac forward bind: `127.0.0.1:18620`
- SSH destination: existing authenticated `<head-spark>` connection over
  Tailscale
- RDMA devices: Mac `rdma_mcrdma1`, Spark `rocep1s0f0`
- Installed Spark daemon:
  `/home/<spark-user>/projects/MCDMA-719219272c9c/install/bin/mcdma-rpcd`
- Mac daemon: pinned `mcdma-rpcd 1.0.0 protocol 1`

Do not add ordinary IPv4 addresses to `mcrdma1` or `enp1s0f0np0`. MCDMA's
README states that its interfaces provide RDMA addressing, not ordinary TCP
networking. The direct link keeps its existing RoCE GIDs and neighbours.

## Compatibility proof

The pinned daemon source uses the configured TCP host and port only to open an
IPv4 stream. It separately opens the selected RDMA device, reads its GID, and
exchanges GID, QP, PSN, rkey, and address values in the `HELLO` messages. The
listen side accepts the TCP connection without using its peer IP to construct
the RDMA path. Therefore the control stream may traverse loopback plus SSH
while RDMA transfers remain on `rdma_mcrdma1` and `rocep1s0f0`.

Relevant pinned source:

- `rpc/rpcd_common.c`: `tcp_connect` opens only the IPv4 TCP stream.
- `rpc/rpcd_connect.c`: `peer_connect` builds and exchanges the RDMA GID/QP
  values independently of that TCP address.
- `rpc/rpcd_listen.c`: the listener constructs its RDMA peer from the `HELLO`
  GID/QP values, not the TCP peer address.
- `docs/link-daemon.md`: only the Mac may reach the control port; shutdown must
  stop the Mac connect daemon before the listen daemon and must not use
  `SIGKILL`.

## Live read-only observations

- Spark SSH forwarding is enabled; gateway forwarding is disabled.
- The established SSH alias authenticates as `<spark-user>` on `<head-spark>`.
- TCP port `18620` was free on both hosts.
- No `mcdma-rpcd` process or SSH forward was running after inspection.
- Explicit loopback binds plus `GatewayPorts=no` prevent remote LAN or tailnet
  clients from reaching the forwarded port.

Local processes on either host could still reach that host's loopback port.
Both machines are owner-controlled; the SSH connection supplies the remote
authentication boundary. No new firewall or Tailscale ACL is needed.

## Required P04B test

P04B must add a bounded helper and isolated loopback fixture that proves:

1. the forward starts only when both loopback ports are free;
2. a Mac loopback request reaches the Spark loopback fixture;
3. neither LAN nor Tailscale address listens on `18620`;
4. closing the SSH control connection closes the forward;
5. no listener, socket, fixture, or SSH process remains;
6. the live GLM API and both containers remain unchanged.

That test must not start either hardware daemon. Hardware startup and transfer
belong to P05.

## Hardware-session order reserved for P05

1. Recheck device/GID indices and port availability.
2. Start the Spark listener on `127.0.0.1:18620`.
3. Start the owner-only SSH local forward on Mac loopback.
4. Start the Mac connector against `127.0.0.1:18620`.
5. Stop the Mac connector with `SHUTDOWN`.
6. Stop the Spark listener with `SHUTDOWN`.
7. Close the SSH control connection.
8. Confirm every process, listener, and Unix socket is gone.

Never use `SIGKILL`; it skips RDMA teardown.
