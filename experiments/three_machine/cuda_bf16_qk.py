"""Run the Spark prefill service with Mac-matching KDA q/k rounding.

This is a deliberately narrow experiment.  It refuses any TensorFold CUDA
source other than the pinned Mia v1.5 image source, patches the three writes
that store normalized q/k values, and builds the result under a distinct
extension name.  The container overlay is discarded when the experiment is
stopped; the pinned image is never modified.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Callable


KDA_MODULE = "tensorfold.families.glm5_next.cuda.kda"
PINNED_KDA_CUDA_SHA256 = (
    "f74056a38e6f18a5d1995cf994e0f3e0231b24a530da827a350f221aedfbdafc"
)
EXPERIMENT_EXTENSION = "tensorfold_glm_kda_v2_bf16qk"
_CHAIN_WRITE = "x[lane * 4 + i] = y;"
_WIDE_WRITE = "dst[base + lane * 4 + i] = y;"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def patch_cuda_source(source: str) -> str:
    """Round the pinned serial CUDA KDA's normalized q/k writes to bfloat16.

    Exact occurrence counts make this fail closed if upstream source changes or
    if an already-patched file is presented.
    """

    if source.count(_CHAIN_WRITE) != 1 or source.count(_WIDE_WRITE) != 2:
        raise RuntimeError(
            "unexpected TensorFold KDA source: expected one serial and two wide "
            "normalized q/k writes"
        )
    patched = source.replace(_CHAIN_WRITE, _CHAIN_WRITE.replace("y;", "bf(y);"))
    patched = patched.replace(_WIDE_WRITE, _WIDE_WRITE.replace("y;", "bf(y);"))
    if patched == source:
        raise RuntimeError("unexpected TensorFold KDA source: patch made no change")
    return patched


def install_kda_variant(
    patcher: Callable[[str], str],
    *,
    extension: str,
    variant: str,
) -> dict[str, Any]:
    """Install one temporary source patch and route KDA to its own JIT build."""

    if KDA_MODULE in sys.modules:
        raise RuntimeError(f"TensorFold KDA loaded before the {variant} experiment patch")
    spec = importlib.util.find_spec(KDA_MODULE)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"cannot locate {KDA_MODULE}")
    source_path = Path(spec.origin).with_name("kda.cu")
    original_bytes = source_path.read_bytes()
    original_sha256 = _sha256(original_bytes)
    if original_sha256 != PINNED_KDA_CUDA_SHA256:
        raise RuntimeError(
            "unexpected TensorFold KDA source hash: "
            f"wanted {PINNED_KDA_CUDA_SHA256}, got {original_sha256}"
        )
    patched_text = patcher(original_bytes.decode("utf-8"))
    patched_bytes = patched_text.encode("utf-8")
    source_path.write_bytes(patched_bytes)

    kda = importlib.import_module(KDA_MODULE)

    @lru_cache(maxsize=1)
    def experimental_extension():
        from tensorfold.cuda.build import load

        here = Path(kda.__file__).parent
        return load(
            name=extension,
            sources=[str(here / "kda.cpp"), str(here / "kda.cu")],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            verbose=False,
        )

    kda._ext = experimental_extension
    event = {
        "event": "glm_cuda_kda_variant",
        "extension": extension,
        "original_sha256": original_sha256,
        "patched_sha256": _sha256(patched_bytes),
        "source": str(source_path),
        "variant": variant,
    }
    print(json.dumps(event, sort_keys=True), flush=True)
    return event


def install_bf16_qk_variant() -> dict[str, Any]:
    """Install the normalized-q/k-only diagnostic variant."""

    return install_kda_variant(
        patch_cuda_source,
        extension=EXPERIMENT_EXTENSION,
        variant="bf16-qk",
    )


def main() -> int:
    install_bf16_qk_variant()
    from experiments.three_machine import cuda_prefill

    return cuda_prefill.main()


if __name__ == "__main__":
    raise SystemExit(main())
