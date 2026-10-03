# Offline autoresearch

This is a small safety checker for proposed experiments. It does not run a
model, contact a machine, or change a service.

## The rule

1. Freeze one baseline file. It names the settings, correct-answer checks, and
   the one speed measurement used for every trial.
2. Copy a trial file and change exactly one setting.
3. Put the already-observed results in that trial file.
4. Validate, then evaluate it. Correctness and safety are checked before speed.
5. Keep only a correct, safe trial that meets the frozen speed target. Reject
   everything else. Stop further work after an OOM, service loss, cleanup
   failure, or recovery failure.

The evaluator rejects a wrong answer, no setting change, more than one setting
change, OOM, service loss, cleanup failure, recovery failure, or a missed speed
target. It never invents or collects measurements.

## Commands

Run these from the repository root:

```sh
python3 -m experiments.autoresearch.cli validate \
  --baseline experiments/autoresearch/fixtures/baseline.json \
  --trial experiments/autoresearch/fixtures/keep.json

python3 -m experiments.autoresearch.cli evaluate \
  --baseline experiments/autoresearch/fixtures/baseline.json \
  --trial experiments/autoresearch/fixtures/keep.json \
  --receipts /tmp/autoresearch-receipts

python3 -m unittest experiments.autoresearch.test_evaluator
```

A kept trial exits with code 0. A rejected trial exits with code 2. An invalid
file exits with code 3. The receipt filename is the SHA-256 of its exact
contents. Running the same input again returns the same file and does not
rewrite it.

## Boundaries

OpenResearch 0.2.14 and OpenCode 1.18.34 live on the Windows controller.
OpenResearch project `a2f3b60c-cebf-4999-bdba-715a45ee1a41` may display and
record these results, but this Python tool does not depend on it. The controller
reached the current GLM endpoint and returned exact `CONTROLLER_READY`; no live
model change was made. These facts are recorded in the sanitized
[OPT1 Windows controller receipt](../results/receipts/OPT1-windows-openresearch-controller-2026-10-01.md).

The method is adapted from Karpathy autoresearch commit
`228791fb499afffb54b46200aca536f79142f117`: fixed evaluation, one variable per
trial, keep/reject decisions, and a durable experiment log. Its nanochat trainer
is excluded because it is a single-NVIDIA-GPU training loop, the Windows
desktop has no usable NVIDIA GPU, and this project is distributed inference.
