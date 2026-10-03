# Portable two-Spark startup receipt

Date: 2026-10-02 UTC

## Outcome

The first same-checkpoint reference launch failed before loading the model.
Both Spark ranks exited with code 1 and neither was killed for memory pressure.
TensorFold reported that `--mtp-confidence` is not a valid control for
GLM-5.3-Flash on its CUDA engine.

This proves the failure was a command-line compatibility error, not a model,
memory, network, or MCDMA performance result. The rejected option was removed
from the CUDA reference command. The number of MTP drafts remains pinned; the
Mac decoder's own draft controls are unchanged.

The failed `PORTABLE-NATIVE-001` raw log is
`results/raw/20261002T225043Z-PORTABLE-NATIVE-001.log`; its SHA-256 is
`058760122c57de7a18a839ff23dd574f573fa60cb0150f8648eea9f365e7e3e0`.

## Reliability changes

Failed native-reference starts now preserve both rank logs and container state
before removing their containers. The three full model hashes also run in
parallel, while retaining the same exact equality and pinned-hash gates. This
reduces repeated startup-check time without weakening the check.

## Remaining gate

The corrected reference must become healthy, complete the fixed benchmark, and
be stopped cleanly. The full Spark-prefill/MCDMA/Mac-decode path must then run
the identical workload and exceed its median capacity by at least 3% while
meeting the project's output-equality and three-machine participation checks.
