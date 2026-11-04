"""Model architecture -> KV-cache geometry and compute cost.

KVC-TierSim's stock presets use one constant (2 MB per 16-token block, 2*P
FLOPs per token) for every model. That is exactly Llama-3-8B in bf16 and is off
by 4x (Llama-2-7B, MHA) to 0.03x (gpt-oss, sliding window) for other models, so
this module derives both quantities from the architecture instead.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache

import yaml

from .paths import CATALOG as _CATALOG_DIR
CATALOG = os.path.join(_CATALOG_DIR, "models.yaml")

DTYPE_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "int4": 0.5, "int2": 0.25}


@dataclass(frozen=True)
class ModelArch:
    key: str
    hf: str
    params_b: float
    active_b: float
    layers: int
    hidden: int
    q_heads: int
    kv_heads: int
    head_dim: int
    attn: str
    ctx: int
    family: str = "other"
    mla: dict | None = None
    swa: dict | None = None
    attn_layers: int | None = None
    ssm_state_mb: float = 0.0
    rank: int | None = None           # survey rank (catalog sync)
    papers: tuple = ()                # ids in catalog/papers.yaml (catalog sync)

    # ------------------------------------------------------------------ layers
    @property
    def kv_layers(self) -> int:
        return self.attn_layers if self.attn == "hybrid-ssm" and self.attn_layers else self.layers

    def _layer_split(self) -> tuple[int, int]:
        """(global_layers, local_window_layers) for hybrid sliding-window models."""
        if self.attn != "hybrid-swa" or not self.swa:
            return self.kv_layers, 0
        r = self.swa["local_per_global"]
        glob = math.ceil(self.layers / (r + 1))
        return glob, self.layers - glob

    # --------------------------------------------------------------- KV bytes
    def kv_bytes_per_token_per_layer(self, dtype: str = "bf16") -> float:
        b = DTYPE_BYTES[dtype]
        if self.attn == "mla":
            return (self.mla["kv_lora_rank"] + self.mla["rope_dim"]) * b
        return 2 * self.kv_heads * self.head_dim * b

    def kv_bytes(self, tokens: int, dtype: str = "bf16") -> float:
        """Resident KV bytes for one sequence of `tokens` tokens (+ SSM state)."""
        per = self.kv_bytes_per_token_per_layer(dtype)
        glob, local = self._layer_split()
        win = self.swa["window"] if self.swa else 0
        total = per * (glob * tokens + local * min(tokens, win))
        return total + self.ssm_state_mb * 1e6 * (DTYPE_BYTES[dtype] / 2.0)

    def kv_bytes_per_token(self, dtype: str = "bf16", tokens: int | None = None) -> float:
        """Average bytes/token. For sliding-window models this depends on length."""
        if tokens is None:
            glob, local = self._layer_split()
            return self.kv_bytes_per_token_per_layer(dtype) * (glob + local)
        return self.kv_bytes(tokens, dtype) / max(tokens, 1)

    def block_bytes(self, tokens_per_block: int = 16, dtype: str = "bf16") -> float:
        """Bytes of one *reusable* prefix block.

        Prefix-cache systems store full-attention KV per block; sliding-window
        layers only matter for the last `window` tokens, so a cached prefix
        block carries global-layer KV only (plus the window tail, amortised
        away for long prefixes). SSM state is per-sequence, not per block.
        """
        glob, _ = self._layer_split()
        return self.kv_bytes_per_token_per_layer(dtype) * glob * tokens_per_block

    # ---------------------------------------------------------------- compute
    def weight_bytes(self, dtype: str = "bf16") -> float:
        return self.params_b * 1e9 * DTYPE_BYTES[dtype]

    def prefill_flops(self, new_tokens: int, cached_tokens: int = 0) -> float:
        """FLOPs to prefill `new_tokens` after `cached_tokens` of reused context.

        Linear part: 2 * active params per token. Attention part: QK^T and AV
        each cost 2*q_heads*d per (query, key) pair, causal over the new chunk
        plus full attention to the cached prefix. MLA attends in latent space
        (d_qk = rank + rope, d_v = rank) when absorbed; sliding-window layers
        cap the key range at the window.
        """
        n, c = float(new_tokens), float(cached_tokens)
        linear = 2.0 * self.active_b * 1e9 * n
        if self.attn == "mla":
            d_qk = self.mla["kv_lora_rank"] + self.mla["rope_dim"]
            d_v = self.mla["kv_lora_rank"]
        else:
            d_qk = d_v = self.head_dim
        per_pair = 2.0 * self.q_heads * (d_qk + d_v)
        pairs_full = n * c + n * (n + 1) / 2.0
        glob, local = self._layer_split()
        if local:
            w = float(self.swa["window"])
            pairs_local = n * min(w, n + c)  # upper bound; fine at block granularity
            attn = per_pair * (glob * pairs_full + local * min(pairs_full, pairs_local))
        else:
            attn = per_pair * self.kv_layers * pairs_full
        return linear + attn

    def decode_bytes_per_step(self, batch_kv_bytes: float, dtype: str = "bf16") -> float:
        """HBM bytes read per decode iteration: active weights + live KV."""
        return self.active_b * 1e9 * DTYPE_BYTES[dtype] + batch_kv_bytes


@lru_cache(maxsize=1)
def load_models(path: str = CATALOG) -> dict[str, ModelArch]:
    raw = yaml.safe_load(open(path))
    out = {}
    for key, v in raw.items():
        if v.get("status") == "pending":  # surveyed, KV geometry not catalogued yet
            continue
        out[key] = ModelArch(key=key, **dict(v, papers=tuple(v.get("papers") or ())))
    return out


def to_kvctiersim_preset(m: ModelArch, tokens_per_block: int = 16, dtype: str = "bf16", gpu_eta: float = 0.5) -> dict:
    """Emit a hardware/models.yaml entry for stock KVC-TierSim.

    Stock KVC-TierSim models recompute as 2*params FLOPs/token; we pass the
    *active* parameter count so MoE models are not overcharged. Use
    kvwl.replay for context-dependent attention cost.
    """
    return {
        "name": m.key,
        "params": m.active_b * 1e9,
        "tokens_per_block": tokens_per_block,
        "bytes_per_block": m.block_bytes(tokens_per_block, dtype),
        "gpu_eta": gpu_eta,
    }
