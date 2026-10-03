"""Real-device P07A correctness fixture, with no network or service startup."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import resource
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from window import Window


def raw(array):
    mx.eval(array)
    mx.synchronize()
    return np.array(array.view(mx.uint16) if array.dtype == mx.bfloat16 else array).tobytes()


def digest(array):
    return hashlib.sha256(raw(array)).hexdigest()


def state(window):
    return {"conv": digest(window.cache.conv), "ssm": digest(window.cache.ssm), "offset": window.cache.offset}


def describe(array):
    return {"sha256": digest(array), "shape": list(array.shape), "dtype": str(array.dtype), "bytes": array.nbytes}


def difference(left, right):
    a, b = np.array(left.astype(mx.float32)), np.array(right.astype(mx.float32))
    delta = a - b
    assert np.isfinite(a).all() and np.isfinite(b).all(), "non-finite layer output"
    return {"exact": raw(left) == raw(right), "max_abs": float(np.max(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta)))}


def run(path):
    started = time.monotonic()
    mx.reset_peak_memory()
    window = Window(path)
    active_loaded = mx.get_active_memory()
    zero = state(window)
    assert window.cache.offset == 0
    assert bool(mx.all(window.cache.conv == 0).item()) and bool(mx.all(window.cache.ssm == 0).item())
    # These IDs deliberately require no tokenizer and are shared with the future CUDA fixture.
    tokens = [((i * 7919 + 17) % 154880) for i in range(32)]
    results = {}
    for rows in (1, 4, 16, 17, 32):
        window.reset()
        first = window.forward_window(tokens[:rows])
        first_state = state(window)
        window.commit(rows)
        assert state(window) == first_state
        window.reset()
        second = window.forward_window(tokens[:rows])
        assert digest(first) == digest(second) and first_state == state(window), "reset/repeat differs"
        window.commit(rows)
        results[str(rows)] = {"output": describe(first), "state": first_state}

    # Short decode arithmetic is advertised row-exact by TensorFold. Wider prefill
    # uses different kernels: report its measured difference, do not invent a tolerance.
    comparisons = {}
    for rows in (4, 16, 17, 32):
        window.reset()
        batch = window.forward_window(tokens[:rows])
        batch_state = window.snapshot()
        window.commit(rows)
        window.reset()
        chunks = []
        for token in tokens[:rows]:
            chunks.append(window.forward_window([token]))
            window.commit(1)
        chunked = mx.concatenate(chunks)
        comparisons[str(rows)] = {
            "output": difference(batch, chunked),
            "conv": difference(batch_state[0], window.cache.conv),
            "ssm": difference(batch_state[1], window.cache.ssm),
            "offset_equal": batch_state[2] == window.cache.offset,
        }
        assert comparisons[str(rows)]["offset_equal"], f"batch/chunk offset mismatch at {rows} rows"
        if rows <= 16:
            assert all(comparisons[str(rows)][part]["exact"] for part in ("output", "conv", "ssm")), \
                f"TensorFold row-exact claim failed at {rows} rows: {comparisons[str(rows)]}"

    replays = {}
    for rows in (4, 17):
        for keep in (0, rows // 2, rows):
            window.reset()
            window.forward_window(tokens[:3])
            window.commit(3)
            entry = state(window)
            window.forward_window(tokens[3:3 + rows])
            window.commit(keep)
            committed = state(window)
            if keep == 0:
                assert committed == entry, "zero commit corrupted entry cache"
            tail = window.forward_window([4242])
            replay_tail, replay_state = describe(tail), state(window)
            window.commit(1)
            window.reset()
            window.forward_window(tokens[:3])
            window.commit(3)
            if keep:
                window.forward_window(tokens[3:3 + keep])
                window.commit(keep)
            assert state(window) == committed, f"commit({keep}) state differs from accepted-prefix replay"
            expected_tail = window.forward_window([4242])
            assert describe(expected_tail) == replay_tail and state(window) == replay_state
            window.commit(1)
            replays[f"{rows}/{keep}"] = {"committed": committed, "tail": replay_tail}

    # Entry-point fail-closed behavior must not advance state.
    window.reset()
    for bad in ([], [-1], [154880], [True]):
        try:
            window.forward_window(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid tokens accepted")
        assert state(window) == zero
    window.forward_window([17])
    for bad in (-1, 2, True):
        try:
            window.commit(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid commit accepted")
    window.commit(0)
    assert state(window) == zero
    paths = window.paths
    assert any(p["fused_hc"] and p["fused_kda"] for p in paths), "short fused path not exercised"
    assert any(not p["decode"] and not p["fused_hc"] and not p["fused_kda"] for p in paths)
    record = {
        "result": "PASS", "tokens": tokens, "tensor_count": len(window.reads),
        "checkpoint": window.identity, "zero_state": zero, "windows": results,
        "batch_vs_single_rows": comparisons, "commits": replays,
        "execution_paths": [dict(t) for t in {tuple(sorted(p.items())) for p in paths}],
        "versions": {n: importlib.metadata.version(n) for n in ("mlx", "mlx-metal", "mlx-lm", "numpy")},
        "memory": {"active_after_load_bytes": active_loaded, "peak_mlx_bytes": mx.get_peak_memory(),
                   "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "elapsed_seconds": time.monotonic() - started,
    }
    record["execution_paths"].sort(key=lambda p: p["rows"])
    window.reset()
    del window
    gc.collect()
    mx.clear_cache()
    mx.synchronize()
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint), indent=2, sort_keys=True))
