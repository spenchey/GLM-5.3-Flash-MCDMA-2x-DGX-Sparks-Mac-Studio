# P04 three-machine readiness recheck

Date: 2026-10-01

Linear issue: `MOT-3332`

Result: **PASS**

This recheck follows the original failed readiness receipt without changing
it. P04A and P04B resolved the missing Linux daemon build and undefined control
endpoint. Neither hardware daemon was started; live hardware operation belongs
to P05.

## Exact daemon placement

| Endpoint | Binary and build | Owner | Control endpoint |
|---|---|---|---|
| Mac Studio | `/Users/<studio-user>/Projects/MCDMA/build/mcdma-rpcd`; `mcdma-rpcd 1.0.0 protocol 1`; SHA-256 `950c7cace2b30c8d45070d61938f4f7b6d80496ab961b4dbf4ed208e4645658b` | `<studio-user>:staff`, mode `0755`; connector runs as `<studio-user>` | SSH local forward on `127.0.0.1:18620` |
| Head Spark | `/home/<spark-user>/projects/MCDMA-719219272c9c/install/bin/mcdma-rpcd`; `mcdma-rpcd 1.0.0 protocol 1`; SHA-256 `dd56f9f58242dc825cf8e9fefcbb4e99640afb76cc729d2a55ece0f4808acc1a` | `<spark-user>:<spark-user>`, mode `0755`; listener runs as `<spark-user>` | listener on `127.0.0.1:18620` only |

The authenticated SSH control-tunnel fixture passed end to end, including
loopback-only exposure, fail-closed stop, conflict refusal, and complete
cleanup. `results/receipts/P04B-control-tunnel-2026-10-01.md` contains the
test evidence.

## Live preservation recheck

- Head GLM container: `running=true`, `oom=false`, restart count `0`.
- Worker GLM container: `running=true`, `oom=false`, restart count `0`.
- Head model API: HTTP `200` and the pinned GLM model remains present.
- No MCDMA daemon, fixture, SSH tunnel, listener, control socket, or temporary
  state remains.

The duplicate Spark machine ID remains an identity-hygiene warning. Distinct
hostnames, boot identities, management/Tailscale addresses, RDMA devices, and
rank placement prevent host conflation. It is recorded and was not changed.

## Decision

P04 now passes: all required OS/software/resource/link observations exist, the
driver and both daemon builds are exact, the future process owners are fixed,
and the control address and port are unambiguous and isolated. P05 must still
prove the real daemon and byte transfers; this readiness pass does not claim
that hardware result.
