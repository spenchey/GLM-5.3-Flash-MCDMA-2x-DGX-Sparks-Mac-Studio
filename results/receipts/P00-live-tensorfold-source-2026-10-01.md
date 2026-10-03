# P00 live TensorFold source receipt

Date: 2026-10-01
Linear issue: `MOT-3327`
Result: **PASS**

P00 captured and reproduced the exact patched TensorFold source used by the
running two-Spark GLM service. All inspection was read-only. Neither container
was stopped, restarted, replaced, or reconfigured.

## Live runtime identity

Both Sparks reported the same image:

- image tag: `tensorfold-glm53:v0.5.0`
- image ID: `sha256:67e82cade069474645782275e2bec5326e91fa0a886831adab8d490f41bbff3f`
- registry digest: `sha256:7bcbb617b1f40f1ce12d37e5ba5e42caf1b1c53b444be9951915e966314c3e1d`
- image size: `24570046863` bytes
- patch label: `cefe8bf45d07`
- TensorFold package version: `0.5.0`
- TensorFold install source: `https://github.com/ashhart/TensorFold.git`
- TensorFold base commit from installed `direct_url.json`:
  `9cd52ab4daba68ddd09be89be8f23ad43175e821`

The public build recipe was clean at commit
`ed026ef92d1650120dada1294a112acb6c8f2f48`, tree
`c25c05dc72a136d86d6e1a71b48b5146d7926f3d`. Its 52 patch files plus the
declared image extras reproduce the image's `cefe8bf45d07` patch label.

## Exact live commands

Head (`<head-spark>`, rank 0):

```text
/opt/nvidia/nvidia_entrypoint.sh tensorfold serve /root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/9eaebb7c4e96d983dcd538e18624622ba5b820a8 --tp 2 --rank 0 --master <private-ip> --master-port 29551 --name GLM-5.3-Flash-EXL3 --host 0.0.0.0 --port 8888 --drafter none --context 1048576 --parallel 1 --thinking --vision
```

Worker (`<worker-spark>`, rank 1):

```text
/opt/nvidia/nvidia_entrypoint.sh tensorfold serve /root/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/9eaebb7c4e96d983dcd538e18624622ba5b820a8 --tp 2 --rank 1 --master <private-ip> --master-port 29551 --drafter none --context 1048576 --parallel 1 --thinking --vision
```

The launch recipe's local selection is `DRAFTER=mtp`; its start command uses
TensorFold's built-in MTP path, represented by `--drafter none` in the live
container command.

## Model identity

- model: `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`
- revision: `9eaebb7c4e96d983dcd538e18624622ba5b820a8`
- host path on each Spark:
  `/home/<spark-user>/.cache/huggingface/hub/models--Mia-AiLab--GLM-5.3-Flash-EXL3-TR3-4bpw/snapshots/9eaebb7c4e96d983dcd538e18624622ba5b820a8`
- size on each Spark: `164G`
- files on each Spark: `144`
- filename-plus-size manifest SHA-256 on each Spark:
  `bba782e0ddcae45440e3fa595dd1e5bf6b7161bf559c94aaa72dfd6ce7b5ad78`

The two complete model manifests compared byte-for-byte equal.

## Extracted source proof

The installed package was extracted from the head and its non-cache files were
hashed. The same manifest was generated independently inside the worker.

- TensorFold source files: `402`
- head source manifest SHA-256:
  `aacb31df395dbb4fed6bd01929b47e1d3d6040b953e8ce457d2fc984303c016c`
- worker source manifest SHA-256:
  `aacb31df395dbb4fed6bd01929b47e1d3d6040b953e8ce457d2fc984303c016c`
- package plus distribution metadata files: `413`
- head and worker install-manifest SHA-256:
  `ff3b1829b07f42706b7ba74d74c126d96997ceb4898edac30695e50b493980d5`

An offline reconstruction then checked out TensorFold base commit
`9cd52ab4daba68ddd09be89be8f23ad43175e821` and applied all 52 patches from
recipe commit `ed026ef92d1650120dada1294a112acb6c8f2f48`. Its installed-file
manifest was exactly equal to the live source manifest above. The only files
present in the reconstructed repository tree but intentionally absent from the
installed package were three README files excluded by packaging.

The ignored evidence bundle is at
`.state/P00-live-source-20261001T170528Z/`. It contains the extracted source,
both host manifests, the model manifests, the public recipe patches, and the
offline reconstruction. Important bundle hashes:

- recipe patch archive:
  `cff233d829af7655881d14d03601712d55a9b8dbf38da6f03c6ceb33fc63e1da`
- generated live-versus-base patch:
  `2f00a0aa26d1a4274d36a27901e04dba8fee7013e08dd402695192f81502eed7`

## Verification commands and observed results

```bash
./scripts/status.sh
./tests/test-config.sh
```

Before inspection and again after all capture work:

- head container: `running=true`, `oom=false`, restart count `0`;
- worker container: `running=true`, `oom=false`, restart count `0`;
- API returned model `GLM-5.3-Flash-EXL3`, owned by TensorFold;
- the Studio MCDMA interface remained up and running; and
- the repository config test printed
  `PASS: pinned inputs and private-data guard`.

Additional equality checks returned no differences:

```bash
diff -u .state/P00-live-source-20261001T170528Z/head-tensorfold-source.sha256 \
  .state/P00-live-source-20261001T170528Z/worker-tensorfold-source.sha256

diff -u .state/P00-live-source-20261001T170528Z/reconstructed-installed-files.sha256 \
  .state/P00-live-source-20261001T170528Z/head-tensorfold-source.sha256

diff -u .state/P00-live-source-20261001T170528Z/head-model-files.txt \
  .state/P00-live-source-20261001T170528Z/worker-model-files.txt
```

P00's pass condition is fully satisfied: the base, patch set, image, installed
files, launch commands, model revision, and two-host equality are now pinned
with independent evidence while the live service remained healthy.
