"""Run Spark prefill with the Mac's Q4 dequantization arithmetic.

This experiment changes one operation only: the CUDA prompt kernel converts
the packed Q4 value, scale, and bias to float32, computes q * scale + bias,
then rounds once to bfloat16.  MLX uses that order on the Mac.  The patch is
applied inside a disposable container overlay and is built under a distinct
extension name, so the pinned TensorFold image is never modified.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


QMM_MODULE = "tensorfold.cuda.kernels.qmm"
PINNED_QMM_PREFILL_SHA256 = (
    "be112df7a73c47530f998caaf44fd75a8490e59ec1152f89d170d72f10c13f9b"
)
EXPERIMENT_EXTENSION = "tensorfold_qmm_v4_metal_dequant"
_BF16_FMA2 = '''__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}'''
_FLOAT32_FMA2 = '''__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
    const float2 av = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(&a));
    const float2 bv = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(&b));
    const float2 cv = __bfloat1622float2(
        *reinterpret_cast<const __nv_bfloat162*>(&c));
    const __nv_bfloat162 d = __floats2bfloat162_rn(
        fmaf(av.x, bv.x, cv.x),
        fmaf(av.y, bv.y, cv.y));
    return *reinterpret_cast<const uint32_t*>(&d);
}'''


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_cuda_source(source: str) -> str:
    """Replace exactly the pinned BF16 dequantizer and otherwise fail closed."""

    if source.count(_BF16_FMA2) != 1 or _FLOAT32_FMA2 in source:
        raise RuntimeError(
            "unexpected TensorFold QMM source: expected one unmodified BF16 fma2"
        )
    patched = source.replace(_BF16_FMA2, _FLOAT32_FMA2)
    if patched == source:
        raise RuntimeError("unexpected TensorFold QMM source: patch made no change")
    return patched


def install_metal_qmm_variant() -> dict[str, Any]:
    """Patch the prompt Q4 kernel and route QMM to a separate JIT build."""

    if QMM_MODULE in sys.modules:
        raise RuntimeError("TensorFold QMM loaded before the metal-qmm experiment patch")
    spec = importlib.util.find_spec(QMM_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {QMM_MODULE}")
    source_path = Path(spec.origin).with_name("qmm_prefill.cu")
    original_bytes = source_path.read_bytes()
    original_sha256 = _sha256(original_bytes)
    if original_sha256 != PINNED_QMM_PREFILL_SHA256:
        raise RuntimeError(
            "unexpected TensorFold QMM source hash: "
            f"wanted {PINNED_QMM_PREFILL_SHA256}, got {original_sha256}"
        )
    patched_bytes = patch_cuda_source(original_bytes.decode("utf-8")).encode("utf-8")
    source_path.write_bytes(patched_bytes)

    qmm = importlib.import_module(QMM_MODULE)

    @lru_cache(maxsize=1)
    def experimental_extension():
        from tensorfold.cuda.build import load

        here = Path(qmm.__file__).parent
        return load(
            name=EXPERIMENT_EXTENSION,
            sources=[
                str(here / "qmm.cpp"),
                str(here / "qmm.cu"),
                str(here / "qmm_prefill.cu"),
                str(here / "qmm_prefill8.cu"),
            ],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )

    qmm._ext = experimental_extension
    event = {
        "event": "glm_cuda_qmm_variant",
        "extension": EXPERIMENT_EXTENSION,
        "original_sha256": original_sha256,
        "patched_sha256": _sha256(patched_bytes),
        "source": str(source_path),
        "variant": "metal-qmm-dequant",
    }
    print(json.dumps(event, sort_keys=True), flush=True)
    return event


def main() -> int:
    install_metal_qmm_variant()
    from experiments.three_machine import cuda_prefill

    return cuda_prefill.main()


if __name__ == "__main__":
    raise SystemExit(main())
