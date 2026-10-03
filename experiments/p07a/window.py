"""Test-only TensorFold embedding + layer [0,1); no final model conclusion."""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from identity import expected, verify
from tensorfold.families.glm5_next import config as C
from tensorfold.families.glm5_next.caches import KDACache
from tensorfold.families.glm5_next.config import Config
from tensorfold.families.glm5_next.weights import Weights, load_layer
from tensorfold.kernels.glm.flash.v1 import hc as HCK
from tensorfold.kernels.glm.flash.v1 import kernels as K


def copy_array(array):
    """Force independent storage across MLX's in-place recurrent-state kernels."""
    mx.eval(array)
    mx.synchronize()
    dtype = array.dtype
    storage = array.view(mx.uint16) if dtype == mx.bfloat16 else array
    result = mx.array(np.array(storage, copy=True))
    result = result.view(dtype) if dtype == mx.bfloat16 else result
    mx.eval(result)
    mx.synchronize()
    return result


class AuditedWeights(Weights):
    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.reads = set()

    def get(self, name):
        assert name in expected(), f"forbidden tensor load: {name}"
        self.reads.add(name)
        return super().get(name)


class Window:
    def __init__(self, path: Path):
        self.identity = verify(path)
        self.cfg = Config.from_dict(json.loads((path / "config.json").read_text()))
        weights = AuditedWeights(path, mtp_layer=self.cfg.num_hidden_layers)
        self.embed = weights.q("embed_tokens")
        mx.eval(*self.embed.arrays())
        self.layer = load_layer(weights, 0, self.cfg)
        assert weights.reads == set(expected()), "incomplete tensor load"
        self.reads = sorted(weights.reads)
        weights._cache.clear()
        self.paths = []
        self.reset()

    def reset(self):
        self.cache = KDACache()
        self.cache.conv = mx.zeros((3, 24576), dtype=mx.bfloat16)
        self.cache.ssm = mx.zeros((1, 64, 128, 128), dtype=mx.float32)
        mx.eval(*self.cache.state)
        mx.synchronize()
        self.pending = None

    def snapshot(self):
        return copy_array(self.cache.conv), copy_array(self.cache.ssm), self.cache.offset

    def restore(self, saved):
        self.cache = KDACache()
        self.cache.conv, self.cache.ssm, self.cache.offset = saved

    def _forward(self, tokens):
        ids = mx.array(tokens, dtype=mx.uint32)
        rows = len(tokens)
        e = self.embed
        h = mx.dequantize(e.weight[ids], e.scales[ids], e.biases[ids],
                          group_size=e.group, bits=e.bits)
        x = mx.contiguous(mx.broadcast_to(h[:, None, :], (rows, 4, 4096)))
        decode = rows <= C.DECODE_ROWS
        layer = self.layer
        fused = (decode and "hc" in C.FUSED and K.metal()
                 and HCK.hc_fits(layer.attn_hc, 4096) and HCK.hc_fits(layer.ffn_hc, 4096))
        if fused:
            x, normed, post, comb = HCK.hc_step(x, None, layer.attn_hc, layer.in_norm, layer.eps)
            pending = (layer.attn(normed, [self.cache], (rows,), decode), post, comb)
            x, normed, post, comb = HCK.hc_step(x, pending, layer.ffn_hc, layer.post_norm, layer.eps)
            pending = (layer.mlp(normed, decode), post, comb)
            x = HCK.hc_step(x, pending, None, None, layer.eps)[0]
        else:
            x = layer(x, [self.cache], (rows,), decode)
        assert x.dtype == mx.bfloat16 and x.shape == (rows, 4, 4096)
        result = mx.contiguous(x.reshape(rows, 16384))
        mx.eval(result, *self.cache.state)
        mx.synchronize()
        assert result.nbytes == rows * 32768
        assert self.cache.conv.shape == (3, 24576) and self.cache.conv.dtype == mx.bfloat16
        assert self.cache.ssm.shape == (1, 64, 128, 128) and self.cache.ssm.dtype == mx.float32
        self.paths.append({"rows": rows, "decode": decode, "fused_hc": bool(fused),
                           "fused_kda": bool(decode and C.FUSED_KDA and layer.attn.fused)})
        return result

    def forward_window(self, tokens):
        if self.pending is not None:
            raise RuntimeError("commit or reset the pending window first")
        tokens = list(tokens)
        if not tokens or any(type(t) is not int or t < 0 or t >= self.cfg.vocab_size for t in tokens):
            raise ValueError("expected nonempty in-vocabulary integer token rows")
        saved = self.snapshot()
        try:
            output = self._forward(tokens)
        except Exception:
            self.restore(saved)
            raise
        self.pending = (tokens, saved)
        return output

    def commit(self, keep):
        if self.pending is None:
            raise RuntimeError("no pending window")
        tokens, saved = self.pending
        if type(keep) is not int or not 0 <= keep <= len(tokens):
            raise ValueError("keep is outside pending window")
        if keep < len(tokens):
            self.restore(saved)
            if keep:
                self._forward(tokens[:keep])
        self.cache._replay = None
        self.pending = None
