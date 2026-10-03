# P04 three-machine readiness receipt

Date: 2026-10-01

Linear issue: `MOT-3332`

Result: **FAIL**

P04's pass proof is not fully satisfied. The three intended machines were
identified and their running TensorFold service, resources, software, driver,
and links were observed without changing them. However, the Linux
`mcdma-rpcd` required for the head Spark was not found, and no deployment
configuration selects its private control address or port. Those are required
values, so they are failures rather than inferred defaults.

## Host identity proof

The inspection originated on a separate control Mac. All results below came
from explicit SSH sessions to the named targets.

| Role | Observed identity | OS, kernel, architecture |
|---|---|---|
| Mac model stage | `spencers-Mac-Studio.local`; Computer Name `spencer’s Mac Studio`; `Mac15,14`; Apple M3 Ultra; 256 GB | macOS `27.0`, build `26A428`; Darwin `27.0.0`, `xnu-13432.1.9~1/RELEASE_ARM64_T6031`; `arm64` |
| TensorFold head/rank 0 | `<head-spark>`; NVIDIA DGX Spark; management `<private-ip>`; Tailscale `<private-ip>`; inter-rank `<private-ip>` | Ubuntu `24.04.4 LTS`; Linux `6.17.0-1031-nvidia` (`#31-Ubuntu`, 2026-07-24); `aarch64` |
| TensorFold worker/rank 1 | `<worker-spark>`; NVIDIA DGX Spark; management `<private-ip>`; Tailscale `<private-ip>`; inter-rank `<private-ip>` | Ubuntu `24.04.4 LTS`; Linux `6.17.0-1031-nvidia` (`#31-Ubuntu`, 2026-07-24); `aarch64` |

The two Sparks have different hostnames, boot IDs, management/Tailscale
addresses, and NIC addresses. They unexpectedly report the same Linux machine
ID (`7af66f30966a49b6886e00e2fce4b42f`); unique DMI serials were not readable by
the unprivileged account. This duplicate is recorded as an identity hygiene
failure, but the live endpoints are not conflated.

## Observed versions and builds

### Mac Studio

- Command Line Tools: `/Library/Developer/CommandLineTools`; macOS SDK `27.0`.
- Python `3.14.7`; MLX `0.32.1` in the default Python environment.
- TensorFold: `not-installed` on the Studio. This matches the current state,
  where the Studio is not yet a TensorFold rank, but it is not evidence of the
  future Metal-stage runtime.
- MLX checkpoint present at revision
  `76add2a341a1cd90ad0e86bb69839ea9c35827c6`.
- MCDMA checkout: clean commit
  `719219272c9ce6fc091b4eab6214eac510b5e387` (commit time
  `2026-09-23T10:51:14+01:00`).
- Loaded driver: `org.mcdma.cx5.native` version `0.1.18`, UUID
  `6BB0406E-E8D6-3965-A610-EC6F788AEFC7`; `kmutil` reports it loaded.
- `rdma_ctl status`: `enabled`.
- Driver registry: `MCDMANativeBuild=26A428`, `MCDMALabEnabled=Yes`,
  `MCDMAGidLive=Yes`, `MCDMATransport=hardware RoCE v2; polled RC; static IPv6
  neighbours`.
- Provider `/usr/local/lib/rdma/libmcdma-rdmav34.so`: SHA-256
  `edf9db51fb4663d14a93757106638497571c7fe9ea2d659b042228f3eb3f5a12`.
- `mcdma-rpcd`: release `1.0.0`, protocol `1`, arm64 binary SHA-256
  `950c7cace2b30c8d45070d61938f4f7b6d80496ab961b4dbf4ed208e4645658b`.
- Security observation: SIP is disabled; authenticated-root protection is
  enabled. No security setting was changed.

### Both Spark containers

- Image ID:
  `sha256:67e82cade069474645782275e2bec5326e91fa0a886831adab8d490f41bbff3f`.
- TensorFold `0.5.0`; live patch label `cefe8bf45d07`.
- Python `3.12.3`; PyTorch `2.13.0a0+9186a08b2c.nv26.07`;
  PyTorch CUDA runtime `13.3`; MLX `not-installed`.
- NVIDIA driver `580.173.02`; `nvidia-smi` driver-advertised CUDA `13.0`.
- Container labels report cuDNN `9.24.0.43` and NCCL `2.30.7`.
- Head runtime: Docker `29.1.3` build `f52814d`; containerd `2.2.1`
  commit `dea7da592f5d1d2b7755e3a161be07f43fad8f75`; runc `1.3.4`
  commit `d6d73eb8`.
- Worker runtime: Docker `29.2.1` build `a5c7197`; the same containerd and
  runc builds as the head. The Docker versions differ and are recorded; no
  project pin establishing that this is incompatible was found.
- Head recipe checkout: commit
  `ed026ef92d1650120dada1294a112acb6c8f2f48`, with only the expected private
  untracked `scripts/local.sh`. No recipe checkout was found at the configured
  path on the worker; the worker runs the pinned image directly.

## Capacity at observation

| Host | Disk | Memory |
|---|---|---|
| Mac Studio | `/`: 926 GiB size, 162 GiB available; Data volume 82% used | 274,877,906,944 bytes total (256 GiB); 207.60 GiB strictly free; 228.97 GiB free + inactive + speculative |
| `<head-spark>` | `/`: 3.7 TiB size, 1.4 TiB available, 63% used | 121 GiB total; 1.2 GiB free; 23 GiB available; swap 15 GiB total / 9.1 GiB free |
| `<worker-spark>` | `/`: 3.7 TiB size, 1.7 TiB available, 55% used | 121 GiB total; 11 GiB free; 18 GiB available; swap 15 GiB total / about 15 GiB free |

These are point-in-time host values, not dedicated model memory.

## Link identities and state

- Studio wired device: `rdma_mcrdma1`, interface `mcrdma1`, ConnectX vendor
  `0x15b3`, part `0x1019`, RoCE v2 GID present, port state `4` / physical state
  `5`, interface `UP,RUNNING`, MTU 9000. The native checker returned
  `native_cx5=2 native_cx5_active_ports=1 errors=0`.
- Studio's other device `rdma_mcrdma0` / `mcrdma0` is down. It was not treated
  as the wired endpoint.
- Head Spark's Mac-facing device/interface: `rocep1s0f0` /
  `enp1s0f0np0`, `ACTIVE`, `LinkUp`, Ethernet, 40 Gb/s, MTU 9000.
- Head Spark's inter-rank device/interface: `rocep1s0f1` /
  `enp1s0f1np1`, `ACTIVE`, `LinkUp`, Ethernet, 200 Gb/s, address `<private-ip>`.
- Worker Spark's inter-rank device/interface: `rocep1s0f1` /
  `enp1s0f1np1`, `ACTIVE`, `LinkUp`, Ethernet, 200 Gb/s, address `<private-ip>`.
- Worker `rocep1s0f0` / `enp1s0f0np0` is down/disabled; it is not the
  Studio-facing endpoint in this topology.

## Process ownership, placement, ports, and health

Observed current placement:

- `<head-spark>`: container `glm53-flash-tf`, blank Docker `Config.User`
  (therefore root), rank 0, PID `3383179` at inspection, API bound to
  `0.0.0.0:8888`, TensorFold rendezvous listening on `*:29551`.
- `<worker-spark>`: container `glm53-flash-tf`, blank Docker `Config.User`
  (therefore root), rank 1, PID `3348760` at inspection, connected from
  `<private-ip>` to head `<private-ip>:29551`.
- Mac Studio: no TensorFold, `mcdma-rpcd`, or MCDMA workload process was
  running. The intended future placement is the MLX/Metal model stage and the
  `mcdma-rpcd connect` end.
- Head Spark: no `mcdma-rpcd` process or binary was found under `/home`,
  `/usr/local`, or `/opt`. The intended future placement is the
  `mcdma-rpcd listen` end on the Mac-facing `rocep1s0f0` device.
- The TensorFold ports are explicitly selected: public model API `8888` and
  rank rendezvous `29551` on the head. The MCDMA daemon's private control
  address and port are **unknown / FAIL**: no deployment value is present and
  the documented `18620` is only an example, so it was not inferred.

Current health:

- Both TensorFold containers: `running=true`, `oom=false`, restart count `0`.
  Neither image defines a Docker healthcheck (`health=none`).
- Head `/v1/models`: HTTP success with model `GLM-5.3-Flash-EXL3`, owner
  `tensorfold`.
- Rank connection: established from `<private-ip>` to `<private-ip>:29551`.
- Studio driver is loaded; `rdma_mcrdma1` and the corresponding head port are
  active. No transfer, benchmark, or daemon health claim is made.
- MCDMA daemon service health: **unknown / FAIL**, because it is not deployed
  or running on either endpoint.

## Exact read-only command set

The commands below were issued through noninteractive SSH. Long outputs were
reduced above without changing the reported values.

```text
ssh -S "$HOME/.ssh/cm-studio" -o ControlMaster=no -o BatchMode=yes <mac-studio> '<date; hostname; scutil; sysctl; sw_vers; uname; df; vm_stat; id; who; ps>'
ssh -S "$HOME/.ssh/cm-studio" -o ControlMaster=no -o BatchMode=yes <mac-studio> '<git rev-parse/status/show; find build artifacts; stat; shasum; file; Python package metadata; kmutil; systemextensionsctl; ifconfig; ioreg; lsof; ps>'
ssh -S "$HOME/.ssh/cm-studio" -o ControlMaster=no -o BatchMode=yes <mac-studio> '<xcode-select; xcrun SDK version; rdma_ctl status; csrutil status; kmutil showloaded; ioreg -c MCDMACX5Native; mcdma-rpcd version; cx5-native-check --require-gid; ifconfig -a>'
ssh -o BatchMode=yes <head-spark> '<hostnamectl; uname; os-release; df; free; nvidia-smi; ip; sysfs RDMA state; rdma link; docker inspect/exec; ps; ss; curl /v1/models>'
ssh -o BatchMode=yes <head-spark> "ssh -o BatchMode=yes <spark-user>@<private-ip> '<the same identity, capacity, NVIDIA, RDMA, container, process, and listener checks>'"
ssh -o BatchMode=yes <head-spark> '<docker/containerd/runc versions; ss connections; nested worker runtime and connection checks>'
```

One direct control-Mac attempt to `<spark-user>@<private-ip>` timed out, as expected
for the private inter-rank address. The worker was then inspected through the
head Spark. Each SSH command exited; no tunnel, sampler, daemon, or other
temporary process was left running.

## Pass-proof decision

**Not satisfied.** Required failures are:

1. no observable Linux `mcdma-rpcd` build/revision on the head Spark;
2. no selected private daemon control address or port;
3. daemon health cannot be observed because neither endpoint is running it;
4. the two Sparks have a duplicated Linux machine ID, and unprivileged DMI
   serial proof was unavailable.

No install, stop, start, restart, configuration change, driver load/unload,
route change, cable action, transfer, or benchmark was performed.
