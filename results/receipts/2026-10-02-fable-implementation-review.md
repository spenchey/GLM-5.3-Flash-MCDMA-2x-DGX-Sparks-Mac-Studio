# Fable Ultra implementation review receipt

Date: 2026-10-02

Reviewer: Claude Code model `claude-fable-5-1`, maximum reasoning effort,
read-only access, launched through the required tmux review harness.

## Coverage attempts

The complete implementation packet, a reduced 116 KiB packet, and three
failure-domain packets each reached the original 600-second harness limit
without returning findings. Their structured timeout record has SHA-256
`49858e878b747e4f9af127f520019de0bda19fb11e1715a0b88f6e3f9df9a3de`.
A timeout was recorded as missing coverage, never as approval.

A combined post-repair operations packet was then given a 1,200-second bound.
It also returned only a structured timeout. That record has SHA-256
`3cb6f4de60f9d6669ffbd8b86dc5b7cbb148d5bb90ad347482f26e5af57639e2`
and is likewise recorded as failed review coverage.

The first reduced normal-lifecycle verification ran for about 13 minutes, then
its wrapper exited nonzero with a shell parsing error and no review. The
preserved wrapper output has SHA-256
`0395c9b7796056e551b40af508e0b7ff93746b86fa23417aa9a9686ae2fba40d`.
It is a failed attempt, not approval.

A second reduced attempt moved result parsing out of the prompt-generating
wrapper. It still exited before returning any review, with only the wrapper
diagnostic `at: garbled time`. The preserved output has SHA-256
`14d2cca916c617e1a6c8b4bffebd6b4c6762c224fbee3a602e8d988de03de5e5`.
This is also failed coverage, not approval. The next attempt bypasses only the
broken wrapper layer while retaining Claude Code, Fable, maximum effort, tmux,
no tools, and read-only review.

## Completed recovery and lifecycle review

Session `112c5d40-4282-4615-89c9-d2b64cb12169` completed with exit zero.
The preserved response has SHA-256
`5589366aa70a6c1c360b6ec19cb1bffe9a8d44cc94e730d7856dc39b16804941`.
It found two P1 and four P2 blockers in the pre-repair lifecycle packet:

1. both-ranks-down stop could leak the Mac API;
2. stale PID files could signal an unrelated process;
3. the API port guard could accept a foreign listener;
4. readiness checks were weaker than the runbook claim;
5. failed transport startup could leak owned processes; and
6. start removed logs and matched containers by name alone.

The same review also reported three P3 hardening items: Docker errors were
masked, rank 0 mounted all of host `/tmp`, and SSH lacked bounded timeouts.
All nine supported findings were repaired: exact PID/listener/image ownership,
full readiness rechecks, pre-side-effect cleanup flags, checked cleanup JSON,
preserved failure evidence, Docker daemon checks, an exact socket bind, and
bounded SSH connect/keepalive behavior. The previously omitted recovery script
was added and independently included in the follow-up packet.

## Post-repair reviews

The direct Claude Code fallback was first proved with an exact sanity response,
then used only after the two wrapper failures above. Normal-lifecycle session
`b4a29878-fd36-43f7-bcbe-0d938f543b37` completed with Fable at maximum effort.
Its preserved JSON has SHA-256
`84f7cdf18153ff320f75e06dbede4284c29d888846bfcdc8c219f293c65390b9`.

It found one P1 and seven P2 defects: mutable-tag launch versus pinned-ID
cleanup, a vacuous final API-owner check, stale reused PID records, stopped
old-image containers blocking a refresh, inflight work being reported as
unhealthy, concurrent-start races, a shorter API load deadline than the rest
of startup, and an unchecked socket bind mount. All eight were repaired. The
exact lifecycle packet must now return clean before the remaining recovery,
deployment, protocol/state, and model-runtime reviews can close.

The first lifecycle rerun, session
`e3d21cfb-c565-45e1-8fed-e5e74bee748e`, returned three remaining P2 findings.
Its preserved JSON has SHA-256
`e297d4d920d6672ae28f666df1b5dba337e57ab6542cf3f4c2ca36352309610d`.
Stop could discard stopped-rank logs, start could remove a stopped unpinned
same-name container, and a transient remote inspection failure was silently
treated like an absent rank during failed-start cleanup. The repairs preserve
logs before removal, refuse the unpinned container, retry inspection, and emit
an explicit incomplete-cleanup stop when rank state cannot be proved.

The second full lifecycle rerun reached its 1,800-second bound with exit 143
and zero response bytes. Its timeout marker has SHA-256
`e9900e1816fa92bf3ec1aca6eeb0904f9db31e1b37dbfeb2dcc30667033e5a2e`.
It is REVIEW-011 and provides no coverage. The same unchanged lifecycle code is therefore split into start/ownership and
stop/status packets; both must complete before the lifecycle review closes.
