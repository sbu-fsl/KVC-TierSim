"""Request-stream representation shared by the profiler and the replay simulator.

A *stream* is an ordered list of requests. Each request carries its prompt as a
list of KV blocks (chained hashes, so equal hash <=> equal prefix up to that
block, exactly like vLLM/SGLang prefix caching) plus, for non-prefix reuse
analysis (CacheBlend/EPIC/KVLink style), the id of the reusable *chunk* each
block belongs to and that block's offset inside the chunk.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

BLOCK_TOKENS = 16


@dataclass
class Request:
    rid: int
    session: str
    turn: int
    input_len: int
    output_len: int
    blocks: list[int]                       # chained prefix hashes, one per full/partial block
    chunk_ids: list[int] = field(default_factory=list)   # per block: reusable chunk id or -1
    arrival: float | None = None            # seconds; None -> assigned by the replay
    block_tokens: int = BLOCK_TOKENS
    out_blocks: list[int] = field(default_factory=list)  # chain continued over generated tokens (multi-turn reuse)


def _h(prev: int, payload: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(prev.to_bytes(8, "little") + payload, digest_size=8).digest(), "little")


def chain_blocks(token_ids, block_tokens: int = BLOCK_TOKENS) -> list[int]:
    """Chained block hashes for a token sequence (last partial block included)."""
    arr = np.asarray(token_ids, dtype=np.int64)
    out, prev = [], 0
    for i in range(0, len(arr), block_tokens):
        prev = _h(prev, arr[i:i + block_tokens].tobytes())
        out.append(prev)
    return out


def continue_blocks(prompt_ids, out_ids, block_tokens: int = BLOCK_TOKENS) -> list[int]:
    """Hashes of the blocks completed once `out_ids` are generated after the prompt."""
    full = len(prompt_ids) // block_tokens
    allb = chain_blocks(list(prompt_ids) + list(out_ids), block_tokens)
    return allb[full:] if out_ids else []


def chunk_hash(token_ids) -> int:
    return _h(0x5EED, np.asarray(token_ids, dtype=np.int64).tobytes())


def save_stream(path: str, reqs: list[Request], meta: dict) -> None:
    with gzip.open(path, "wt") as f:
        f.write(json.dumps({"meta": meta}) + "\n")
        for r in reqs:
            d = asdict(r)
            f.write(json.dumps(d, separators=(",", ":")) + "\n")


def load_stream(path: str) -> tuple[dict, list[Request]]:
    reqs, meta = [], {}
    with gzip.open(path, "rt") as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            if i == 0 and "meta" in d:
                meta = d["meta"]
                continue
            reqs.append(Request(**d))
    return meta, reqs


# ----------------------------------------------------------------- statistics
def reuse_stats(reqs: list[Request]) -> dict:
    """Infinite-cache reuse characteristics of a stream (order as given)."""
    seen_blocks: set[int] = set()
    seen_chunks: set[int] = set()
    tot_blocks = hit_blocks = 0
    tot_tokens = hit_tokens = 0
    chunk_tokens_reusable = 0
    for r in reqs:
        bt = r.block_tokens
        n = len(r.blocks)
        # prefix hits stop at the first miss (chained hashes guarantee it anyway)
        k = 0
        while k < n and r.blocks[k] in seen_blocks:
            k += 1
        # only full blocks can be served from cache
        full = r.input_len // bt
        k_full = min(k, full)
        hit_blocks += k_full
        hit_tokens += k_full * bt
        tot_blocks += n
        tot_tokens += r.input_len
        if r.chunk_ids:
            for j in range(k_full, n):
                c = r.chunk_ids[j]
                if c != -1 and c in seen_chunks:
                    chunk_tokens_reusable += bt
            seen_chunks.update(c for c in r.chunk_ids if c != -1)
        seen_blocks.update(r.blocks)
        seen_blocks.update(r.out_blocks)
    return {
        "requests": len(reqs),
        "prefix_hit_ratio": hit_tokens / tot_tokens if tot_tokens else 0.0,
        "nonprefix_extra_ratio": chunk_tokens_reusable / tot_tokens if tot_tokens else 0.0,
        "unique_block_ratio": len(seen_blocks) / tot_blocks if tot_blocks else 0.0,
        "unique_blocks": len(seen_blocks),
        "total_blocks": tot_blocks,
    }


def length_stats(xs) -> dict:
    a = np.asarray(xs, dtype=float)
    if a.size == 0:
        return {}
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "min": float(a.min())}
