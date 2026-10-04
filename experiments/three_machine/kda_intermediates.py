"""Names and rank-merge rules for one diagnostic KDA prompt layer.

These tensors are evidence only.  They are absent from normal cache handoffs
and never participate in decode.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from experiments.three_machine.cache_handoff import HandoffError, require_replicated


DIAGNOSTIC_PREFIX = "diagnostic.kda.layer"
KDA_INTERMEDIATE_STAGES = (
    "input",
    "projection",
    "forget_projection",
    "key",
    "value",
    "decay",
    "beta",
    "output_gate_projection",
    "output",
    "state",
)


def kda_intermediate_name(layer: int, stage: str) -> str:
    if int(layer) < 0 or stage not in KDA_INTERMEDIATE_STAGES:
        raise ValueError("invalid KDA intermediate layer or stage")
    return f"{DIAGNOSTIC_PREFIX}.{int(layer)}.{stage}"


def kda_intermediate_names(layer: int) -> dict[str, str]:
    return {
        stage: kda_intermediate_name(layer, stage)
        for stage in KDA_INTERMEDIATE_STAGES
    }


def merge_kda_projection(
    rank_values: Sequence[np.ndarray],
    *,
    local_heads: int,
    head_dim: int = 128,
) -> np.ndarray:
    """Merge ``q|k|v|f_a|g_a|beta`` projection shards in MLX order.

    CUDA shards q, k, v, and beta by head while replicating f_a and g_a.
    Concatenating complete rank rows would interleave those groups and produce
    a tensor that cannot be compared with the unsharded Metal projection.
    """

    if len(rank_values) != 2 or int(local_heads) <= 0 or int(head_dim) <= 0:
        raise HandoffError("KDA projection merge needs two valid Spark shards")
    values = [np.ascontiguousarray(value) for value in rank_values]
    base = values[0]
    width = int(local_heads) * int(head_dim)
    expected = 3 * width + 2 * int(head_dim) + int(local_heads)
    if base.ndim < 1 or base.shape[-1] != expected:
        raise HandoffError("Spark KDA projection shard has unexpected width")
    if any(value.shape != base.shape or value.dtype != base.dtype for value in values[1:]):
        raise HandoffError("Spark KDA projection shards have incompatible geometry")

    q = [value[..., :width] for value in values]
    k = [value[..., width:2 * width] for value in values]
    v = [value[..., 2 * width:3 * width] for value in values]
    fa = [value[..., 3 * width:3 * width + head_dim] for value in values]
    ga = [value[..., 3 * width + head_dim:3 * width + 2 * head_dim] for value in values]
    beta = [value[..., 3 * width + 2 * head_dim:] for value in values]
    fa_value = require_replicated(fa, name="KDA f_a projection")
    ga_value = require_replicated(ga, name="KDA g_a projection")
    return np.ascontiguousarray(np.concatenate([
        *q,
        *k,
        *v,
        fa_value,
        ga_value,
        *beta,
    ], axis=-1))


def merge_head_shards(
    rank_values: Sequence[np.ndarray],
    *,
    axis: int,
    name: str,
) -> np.ndarray:
    """Join two head-sharded diagnostic arrays after strict shape checks."""

    if len(rank_values) != 2:
        raise HandoffError(f"exactly two Spark shards of {name} are required")
    values = [np.ascontiguousarray(value) for value in rank_values]
    if values[0].ndim == 0:
        raise HandoffError(f"Spark diagnostic {name!r} cannot be scalar")
    normalized_axis = int(axis) % values[0].ndim
    base_shape = list(values[0].shape)
    for value in values[1:]:
        if value.ndim != values[0].ndim or value.dtype != values[0].dtype:
            raise HandoffError(f"Spark diagnostic shards for {name!r} are incompatible")
        shape = list(value.shape)
        if any(
            shape[index] != base_shape[index]
            for index in range(len(shape))
            if index != normalized_axis
        ):
            raise HandoffError(f"Spark diagnostic shards for {name!r} have incompatible shapes")
    return np.ascontiguousarray(np.concatenate(values, axis=normalized_axis))
