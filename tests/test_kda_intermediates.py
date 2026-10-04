import numpy as np
import pytest

from experiments.three_machine.cache_handoff import HandoffError
from experiments.three_machine.kda_intermediates import (
    KDA_INTERMEDIATE_STAGES,
    kda_intermediate_names,
    merge_head_shards,
    merge_kda_projection,
)


def test_kda_intermediate_names_are_complete_and_layer_scoped():
    names = kda_intermediate_names(7)
    assert tuple(names) == KDA_INTERMEDIATE_STAGES
    assert names["input"] == "diagnostic.kda.layer.7.input"
    assert len(set(names.values())) == len(KDA_INTERMEDIATE_STAGES)


def test_kda_projection_merge_restores_unsharded_mlx_order():
    rank0 = np.asarray([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.uint16)
    rank1 = np.asarray([[12, 13, 14, 15, 16, 17, 7, 8, 9, 10, 18]], dtype=np.uint16)

    merged = merge_kda_projection(
        [rank0, rank1],
        local_heads=1,
        head_dim=2,
    )

    assert merged.tolist() == [[
        1, 2, 12, 13,
        3, 4, 14, 15,
        5, 6, 16, 17,
        7, 8,
        9, 10,
        11, 18,
    ]]


def test_kda_projection_merge_refuses_changed_replicated_controls():
    rank0 = np.zeros((2, 11), dtype=np.uint16)
    rank1 = rank0.copy()
    rank1[:, 6] = 1
    with pytest.raises(HandoffError, match="replicated tensor"):
        merge_kda_projection([rank0, rank1], local_heads=1, head_dim=2)


def test_head_shard_merge_checks_non_head_geometry():
    left = np.zeros((3, 2, 4), dtype=np.float32)
    right = np.ones((3, 1, 4), dtype=np.float32)
    assert merge_head_shards([left, right], axis=1, name="key").shape == (3, 3, 4)

    with pytest.raises(HandoffError, match="incompatible shapes"):
        merge_head_shards(
            [left, np.ones((4, 1, 4), dtype=np.float32)],
            axis=1,
            name="key",
        )
