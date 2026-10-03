# TensorFold GLM baseline receipt

Date: 2026-10-01

## Result

PASS: GLM-5.3-Flash-EXL3 is serving through TensorFold on the two DGX Sparks.

## Pinned inputs

- MiaAI recipe commit: `ed026ef92d1650120dada1294a112acb6c8f2f48`
- Model revision: `9eaebb7c4e96d983dcd538e18624622ba5b820a8`
- TensorFold image: `sha256:67e82cade069474645782275e2bec5326e91fa0a886831adab8d490f41bbff3f`
- TensorFold package: `0.5.0`
- TensorFold patch label: `cefe8bf45d07`
- Drafter: checkpoint MTP; DFlash2 excluded
- Context window: 1,048,576 tokens
- KV cache: FP8

## Health proof

- Head container: running, OOM false, restart count 0
- Worker container: running, OOM false, restart count 0
- Model endpoint listed `GLM-5.3-Flash-EXL3`, owned by TensorFold
- A fresh chat request returned exactly `GLM_READY`
- Request usage: 18 prompt tokens, 5 completion tokens
- TensorFold response receipt SHA: `a0c7ccf1cfa0b4e6`

## Transport proof and boundary

- The two-Spark recipe reported RoCE for small all-gathers and NCCL for larger
  transfers.
- The separate Mac Studio MCDMA preflight passed for its pinned driver,
  provider, interface, and Spark endpoint.
- The Mac-to-Spark link currently negotiates at 40 Gb/s.
- This receipt does not claim TensorFold-over-MCDMA. The live GLM request used
  only the two-Spark TensorFold service; the Mac Studio was not a model rank.

## Source-evidence boundary

The live image contains a patched TensorFold build and did not expose Git
metadata. Its exact base commit and patched file hashes have not yet been
captured. Therefore the context and FP8 lines above remain deployment facts to
reconfirm from the running command and extracted live code; they must not be
accepted or rejected solely from the separate upstream research checkout.
