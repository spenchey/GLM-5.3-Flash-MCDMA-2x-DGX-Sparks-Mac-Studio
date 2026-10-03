# Three-machine operator runbook

This is the shortest safe path to operate the verified deployment. Commands
run from the repository root on the control Mac.

## What must already exist

1. The Mac Studio sees the supported ConnectX-5 Ex card and MCDMA interface.
2. The head Spark is cabled to the Studio card; the two Sparks retain their
   separate private high-speed link.
3. Passwordless SSH works from the control Mac to all three machines.
4. Both Sparks have the pinned model snapshot and identical pinned container
   image.
5. The Mac has the pinned stage checkpoint, isolated Python environment,
   TensorFold Metal source, and MCDMA library.
6. `config.local.env` exists locally and is not committed.

No script downloads a replacement for a missing pinned artifact.
To rebuild the Mac TensorFold source from the exact Mia base, use:

```bash
./scripts/prepare-tensorfold-metal-stage.sh /path/to/exact-mia-source /new/destination
```

It rejects a changed base, verifies both patched source files, and verifies the
complete prepared source tree against the committed manifest. Preflight checks
the complete tree and its review marker on the Studio before every start. The
prepared source and stage checkpoint must live outside the deployed runtime
directory so a code update can never replace them.

## Check after the first deployment and before starting

```bash
./scripts/preflight-three-machine.sh
```

On a new host, run deployment once before this check so the three protected
runtime directories and their deployment markers exist. The check must end
with `PASS`. It verifies both Spark images, identical
filename/size and metadata/symlink manifests for the two pinned model copies,
one deployment ID and one code/configuration manifest across all three runtime
hosts, Mac/head clock agreement for absolute frame deadlines, the complete
Mac TensorFold source and stage-directory manifests, required Mac libraries,
and the Metal runtime.

## Start

```bash
./scripts/stop-three-machine.sh  # only when an older copy is already running
./scripts/deploy-three-machine.sh
./scripts/start-three-machine.sh
```

Deployment refuses to replace code while the API or either rank is running.
It also refuses a dirty or uncommitted project. It installs the exact runtime
subset from the current Git commit (`UPSTREAM.lock`, the public configuration
template, scripts, the three-machine implementation, and the TensorFold
patches), plus the ignored private configuration in mode `0600`. It verifies
one code manifest, configuration hash, and unique deployment ID across all
hosts, and retains the immediately prior code directory for exact rollback.
Documentation, tests, model weights, raw evidence, and prepared TensorFold
artifacts are not copied by deployment and are never deleted. The private
configuration is copied only into each protected runtime directory and remains
excluded from Git and the public content manifest.

Use dedicated runtime paths such as
`~/.local/share/glm53-three-machine/runtime`; never point deployment at a Git
checkout, model cache, prepared TensorFold tree, stage checkpoint, MCDMA build,
or process-state directory. Deployment rejects those overlaps.

If deployment stops partway through, keep the service stopped. If the script
reports an incomplete rollback, or the three deployment IDs differ, rerun
`./scripts/deploy-three-machine.sh` from the same clean commit and unchanged
private configuration. The deployment state machine is designed to converge
all three hosts to that one version. Then rerun preflight. Never hand-move or
delete a `.previous` directory because each host can have a different prior
version after an interrupted install.

The start order is deliberate:

1. verify the pinned artifacts;
2. start and prove the MCDMA link;
3. start CUDA rank 1 on the worker Spark;
4. start CUDA rank 0 on the head Spark;
5. wait for both ranks and the MCDMA mailbox;
6. start the persistent Mac API and Metal stage; and
7. require a successful API health check.

If startup fails, the script stops only the experimental containers it owns and
cleans up the MCDMA processes it started. It does not delete model data.

Every start, stop, recovery, and deployment uses one shared lifecycle lock on
the Mac Studio. A later command automatically recovers a lock whose token proves
that the same control Mac's exact owning process is gone. It also recovers an
ownerless lock directory only after the interrupted acquisition has remained
abandoned for more than one minute. If a command still reports that the lock is
held, do not delete it and do not start another copy. First confirm that no
`start-three-machine.sh`, `stop-three-machine.sh`,
`recover-three-machine.sh`, or `deploy-three-machine.sh` process is still
running on the control Mac. Read the lock's owner token with the configured
Studio SSH path and `$THREE_MACHINE_API_STATE_DIR/lifecycle.lock/owner`. Never
manually remove a lock created by a different controller: the Studio cannot
prove that controller has stopped, so the safe action is to restore that
controller or investigate the recorded owner.

## Check health

```bash
./scripts/status-three-machine.sh
```

A healthy result shows:

- head Spark `running=true`, `oom=false`;
- worker Spark `running=true`, `oom=false`;
- MCDMA `link=up`, both sides `alive=true`, `inflight=false`;
- zero MCDMA failures; and
- Mac API `status=ready`.

## Send a request

```bash
./scripts/chat-three-machine.sh "Reply with exactly THREE_MACHINE_READY" 64
```

The API is compatible with `POST /v1/chat/completions` for non-streaming greedy
requests. Use `temperature: 0`. The output-token parameter accepts 1 to 2,051,
but the prompt plus requested output must fit the 2,051-position dense context.
This limit is intentional: the split CUDA engine has no sparse long-context
index state yet. This first version accepts one request at a time.

## Measure it

```bash
./scripts/benchmark-three-machine.sh --repeats 5 --warmups 1 --max-tokens 128
```

The benchmark requires the same fixed output hash in every run. Record the raw
log and its SHA-256 before changing one performance setting. Follow
`docs/UPSTREAM-OPTIMIZATION-PLAN.md`: use drift-aware interleaved A/B rounds,
keep a change only when correctness and all three-machine health stay exact,
and require a gain larger than measured host drift.

### Final concurrent capacity gate

The final proof is a separate, temporary campaign. It does not replace the
single-request API above. It compares the same 16 requests in a fixed paired
order:

- baseline: all 16 queue through the pinned two-Spark TensorFold service;
- candidate: eight use that Spark service while eight decode together on the
  Mac from one MCDMA-transferred prompt cache.

Prepare the immutable cache and exact 256-token answer once:

```bash
./scripts/prepare-concurrent-cache.sh
```

The last line is the prompt digest. Use it for two independent clean runs:

```bash
./scripts/benchmark-concurrent-capacity.sh PROMPT_DIGEST 16 8 256
./scripts/benchmark-concurrent-capacity.sh PROMPT_DIGEST 16 8 256
```

Each run starts fresh native and replay containers, executes warmups followed
by `ABBABAAB`, then stops every owned process. It passes only when every pair,
the median pair, and the conservative lower confidence bound beat the baseline
by at least 3%; first-token and completion tails may regress by no more than
2%. Every lane must return the exact frozen tokens and bytes. Every candidate
round must use the same three cache frames and exactly five MCDMA calls, with
unchanged daemon and container identities throughout.

Compare the two resulting `results/raw/paired-capacity-*` directories:

```bash
python3 -m experiments.three_machine.paired_repeatability FIRST_RUN SECOND_RUN
```

That final check requires the same frozen reference, allocation, and stable
configuration, but new container IDs and start times. This is the evidence that
the result survives a clean restart rather than depending on one warm process.
When raw Docker inspection files are present, the comparator binds them to the
accepted IDs and start times and recomputes the stable settings. It ignores only
the order of unique environment keys, which Docker/NVIDIA may reorder; duplicate
keys keep their order because their order can change the effective value.

## Stop cleanly

```bash
./scripts/stop-three-machine.sh
```

The stop command asks the Mac API to close. The API then sends an orderly
MCDMA shutdown to both Spark ranks, waits for both containers to exit with code
zero, and removes the owned MCDMA connector, listener, and tunnel. No model or
checkpoint file is removed.

## If one part is unhealthy

Do not start a second copy and do not kill unrelated processes.

1. Save `./scripts/status-three-machine.sh` output.
2. Save both experimental container logs.
3. If both ranks are still running, try `./scripts/stop-three-machine.sh` once.
4. If one rank has failed or the API/transport is unhealthy, run:

   ```bash
   ./scripts/recover-three-machine.sh
   ```

5. Recovery first preserves logs and exact state under ignored `results/raw/`,
   verifies that every surviving process/container belongs to this deployment,
   stops only those verified owners, cleans MCDMA, restarts all three machines,
   and requires an exact `THREE_MACHINE_READY` answer.

An exact forced stop is reserved for a verified experimental process or rank
that ignores the bounded shutdown deadline after a partial failure. The
recovery evidence records when this was necessary. It never targets model data
or an unrelated container.

## Proven operating envelope

- Mac: tokenizer, embedding, original GLM layer 0, persistent API.
- Sparks: original GLM layers 1 through 44, final norm, head, greedy token.
- Transport: MCDMA only; no silent fallback.
- Spark-to-Spark synchronization: TensorFold NCCL on the private link.
- Request type: one non-streaming greedy text request.
- Persistent API not enabled: MTP, prefix reuse, concurrency, streaming,
  sampling, vision, tools, structured output, or the complete upstream Mia
  server surface. The temporary concurrent capacity gate above is separately
  supported and torn down after each run.

Those exclusions are explicit so a basic verified service is never mislabeled
as feature parity with the two-Spark Mia recipe.
