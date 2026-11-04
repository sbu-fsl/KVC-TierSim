"""Trace-driven KV-cache replay on top of KVC-TierSim's hardware and policies.

KVC-TierSim answers "restore or recompute?" for ONE request with given hit/miss
block counts. This module produces those counts from a real request stream: it
keeps a block-level, multi-tier (HBM -> DRAM -> SSD) prefix cache, replays the
stream through a prefill queue and a decode pool, and for every request hands
KVC-TierSim's policy the actual residency of its hit blocks.

Modeling assumptions (kept deliberately simple and stated in the README):
  * one serving instance of `gpu_count` GPUs (tensor parallel), FCFS prefill
    queue; prefill and decode do not interfere (disaggregated-style);
  * restore and recompute overlap (layer-wise pipelining), so a request's
    prefill service time is max(restore, recompute) as in KVC-TierSim;
  * decode iteration time = (active weights + live KV bytes) / HBM bandwidth;
  * live (decoding) KV is pinned in HBM and shrinks the HBM prefix cache;
  * evicted HBM blocks are written back to DRAM, DRAM victims to SSD
    (LMCache/Mooncake-style), costing link/SSD write bandwidth.
"""

from __future__ import annotations

import heapq
import math
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, field, asdict

import numpy as np

from .paths import ensure_simulator_importable

ensure_simulator_importable()

from simulator.core.pool import GPUS, DRAMS, DISKS, LINKS            # noqa: E402  (KVC-TierSim)
from simulator.core.policies import PolicyContext, get_policies       # noqa: E402
from simulator.core.policies.core import evaluate                      # noqa: E402

from .models import ModelArch, DTYPE_BYTES                  # noqa: E402
from .streams import Request                                # noqa: E402


# ------------------------------------------------------------------ configs
@dataclass(frozen=True)
class Hardware:
    gpu: str = "H100"
    dram: str = "DDR5"
    disk: str = "NVMe"
    link: str = "PCIe5"
    gpu_count: int = 0            # 0 -> smallest power of two that fits the weights
    dram_count: int = 4
    prefill_eta: float = 0.25     # achieved fraction of the catalog's headline (sparse) FLOPS
    decode_eta: float = 0.7       # achieved fraction of HBM bandwidth
    hbm_reserve: float = 0.10     # activations / fragmentation
    disk_capacity_gb: float = 0   # 0 -> catalog capacity


@dataclass(frozen=True)
class Technique:
    name: str
    prefix_cache: bool = True
    tiers: tuple = ("hbm",)
    eviction: str = "lru"                    # lru | lfu | fifo
    restore_policy: str = "default_policy"   # KVC-TierSim: default_policy | all_compute | performance_aware
    kv_dtype: str = "bf16"                   # bf16 | fp8 | int4 (stored & restored KV)
    dequant_s_per_GB: float = 0.0            # GPU time to decode compressed KV (CacheGen-style codecs)
    nonprefix: bool = False                  # CacheBlend/EPIC chunk reuse
    blend_recompute: float = 0.15            # fraction of reused-chunk tokens recomputed
    kv_budget: int = 0                       # token-eviction budget for decode KV (SnapKV/H2O); 0 = off


TECHNIQUES = {
    "no_cache":        Technique("no_cache", prefix_cache=False),
    "prefix_lru":      Technique("prefix_lru"),
    "prefix_lfu":      Technique("prefix_lfu", eviction="lfu"),
    "prefix_fifo":     Technique("prefix_fifo", eviction="fifo"),
    "tiered_restore":  Technique("tiered_restore", tiers=("hbm", "dram", "disk")),
    "tiered_ioaware":  Technique("tiered_ioaware", tiers=("hbm", "dram", "disk"), restore_policy="performance_aware"),
    "tiered_fp8":      Technique("tiered_fp8", tiers=("hbm", "dram", "disk"), restore_policy="performance_aware", kv_dtype="fp8"),
    "tiered_int4":     Technique("tiered_int4", tiers=("hbm", "dram", "disk"), restore_policy="performance_aware", kv_dtype="int4",
                                 dequant_s_per_GB=0.002),
    "snapkv_2k":       Technique("snapkv_2k", kv_budget=2048),
    "cacheblend":      Technique("cacheblend", tiers=("hbm", "dram", "disk"), restore_policy="performance_aware", nonprefix=True),
}


# ------------------------------------------------------------------ cache
class Tier:
    def __init__(self, name, capacity_blocks, eviction):
        self.name, self.cap, self.ev = name, max(0, int(capacity_blocks)), eviction
        self.od: OrderedDict[int, None] = OrderedDict()
        self.freq: dict[int, int] = {}

    def __contains__(self, k):
        return k in self.od

    def touch(self, k):
        if k in self.od:
            self.freq[k] = self.freq.get(k, 0) + 1
            if self.ev == "lru":
                self.od.move_to_end(k)

    def victim(self):
        if self.ev == "lfu":   # sampled LFU over the 32 oldest entries (Redis-style approximation)
            best, bf = None, None
            for i, k in enumerate(self.od):
                f = self.freq.get(k, 0)
                if bf is None or f < bf:
                    best, bf = k, f
                if i >= 31:
                    break
            return best
        return next(iter(self.od))

    def pop(self, k):
        self.od.pop(k, None)
        self.freq.pop(k, None)


class TieredCache:
    def __init__(self, caps: dict[str, float], eviction: str):
        self.tiers = [Tier(n, c, eviction) for n, c in caps.items()]
        self.writes = {t.name: 0 for t in self.tiers}   # blocks written per tier (demotions)
        self.pinned = 0                                  # HBM blocks held by decoding requests

    def where(self, k):
        for i, t in enumerate(self.tiers):
            if k in t:
                return i
        return None

    def _make_room(self, i):
        t = self.tiers[i]
        cap = t.cap - (self.pinned if i == 0 else 0)
        while len(t.od) > max(0, cap):
            v = t.victim()
            t.pop(v)
            if i + 1 < len(self.tiers):
                if self.where(v) is None:          # not already resident lower down (inclusive copies)
                    self.tiers[i + 1].od[v] = None
                    self.writes[self.tiers[i + 1].name] += 1
                    self._make_room(i + 1)

    def insert_hbm(self, keys):
        h = self.tiers[0]
        for k in keys:
            if k in h.od:
                h.touch(k)
            else:
                h.od[k] = None
                h.freq[k] = h.freq.get(k, 0) + 1
        self._make_room(0)

    def touch(self, k, i):
        self.tiers[i].touch(k)

    def set_pinned(self, blocks):
        self.pinned = blocks
        self._make_room(0)


# ------------------------------------------------------------------ engine
def auto_gpu_count(m: ModelArch, hw: Hardware) -> int:
    if hw.gpu_count:
        return hw.gpu_count
    wd = "fp8" if m.params_b > 200 else "bf16"
    for n in (1, 2, 4, 8, 16, 32):
        if m.weight_bytes(wd) <= 0.65 * n * GPUS[hw.gpu].hbm_capacity:
            return n
    return 32


class Engine:
    def __init__(self, model: ModelArch, hw: Hardware, tech: Technique, block_tokens: int = 16):
        self.m, self.hw, self.t, self.bt = model, hw, tech, block_tokens
        g = GPUS[hw.gpu]
        self.n_gpu = auto_gpu_count(model, hw)
        self.wdtype = "fp8" if model.params_b > 200 else "bf16"
        self.flops = g.peak_flops * self.n_gpu * hw.prefill_eta
        self.hbm_bw = g.hbm_bandwidth * self.n_gpu
        kv_hbm = g.hbm_capacity * self.n_gpu * (1 - hw.hbm_reserve) - model.weight_bytes(self.wdtype)
        self.kv_hbm_bytes = max(kv_hbm, 0.0)
        # bytes per cached block: full-attention layers only (sliding-window layers are not reusable prefix state)
        self.block_bytes_bf16 = model.block_bytes(block_tokens, "bf16")
        self.block_bytes = model.block_bytes(block_tokens, tech.kv_dtype)
        # live (decode) KV is kept in bf16 regardless of the storage codec
        self.live_block_bytes = model.kv_bytes(block_tokens, "bf16") if model.attn != "hybrid-swa" else self.block_bytes_bf16
        caps = {"hbm": self.kv_hbm_bytes / self.block_bytes}
        if "dram" in tech.tiers:
            caps["dram"] = DRAMS[hw.dram].capacity * hw.dram_count / self.block_bytes
        if "disk" in tech.tiers:
            dcap = (hw.disk_capacity_gb * 1e9) if hw.disk_capacity_gb else DISKS[hw.disk].capacity
            caps["disk"] = dcap / self.block_bytes
        self.cache = TieredCache(caps, tech.eviction)
        self.hbm_total_blocks = int(self.kv_hbm_bytes / self.live_block_bytes)
        link = min(LINKS[hw.link].bandwidth, g.effective_link_bandwidth(LINKS[hw.link].bandwidth))
        b = self.block_bytes
        self.x = {  # restore throughput, blocks/s, per tier (KVC-TierSim formulas)
            0: self.hbm_bw / b,
            1: 1.0 / (b / DRAMS[hw.dram].bandwidth + b / link),
            2: 1.0 / (b / DISKS[hw.disk].bandwidth + b / link),
        }
        self.policy = {p.name: p for p in get_policies()}[tech.restore_policy]

    # -- costs
    def prefill_time(self, new_tokens, cached_tokens=0):
        return self.m.prefill_flops(new_tokens, cached_tokens) / self.flops if new_tokens > 0 else 0.0

    def decode_iter_time(self, live_kv_bytes):
        w = self.m.active_b * 1e9 * DTYPE_BYTES[self.wdtype]
        return (w + live_kv_bytes) / (self.hbm_bw * self.hw.decode_eta)

    def kv_blocks_live(self, r: Request):
        tokens = r.input_len + r.output_len
        if self.t.kv_budget:
            tokens = min(tokens, self.t.kv_budget + r.output_len)
        if self.m.attn == "hybrid-swa":
            return math.ceil(self.m.kv_bytes(tokens) / self.live_block_bytes)
        return math.ceil(tokens / self.bt)

    # -- one prefill
    def serve_prefill(self, r: Request, rate: float, slo: float):
        bt = r.block_tokens
        scale = bt / self.bt                                    # trace blocks may be 512 tokens
        full = r.input_len // bt
        hits_by_tier = [0, 0, 0]
        k = 0
        if self.t.prefix_cache:
            while k < full:
                i = self.cache.where(r.blocks[k])
                if i is None:
                    break
                hits_by_tier[i] += 1
                k += 1
        prefix_hit = k
        blend = [0, 0, 0]
        blend_blocks = 0
        if self.t.nonprefix and r.chunk_ids:
            off, prev = 0, None
            for j in range(full):
                c = r.chunk_ids[j]
                off = off + 1 if c == prev else 0
                prev = c
                if j < prefix_hit or c == -1:
                    continue
                i = self.cache.where(hash((c, off)))
                if i is not None:
                    blend[i] += 1
                    blend_blocks += 1
        hit = prefix_hit + blend_blocks
        # residency-blended restore throughput over non-HBM hits (HBM hits are already in place)
        res = [hits_by_tier[i] + blend[i] for i in range(3)]
        n_restore = res[1] + res[2]
        inv = sum(res[i] / self.x[i] for i in (1, 2) if res[i])
        x_eff = (n_restore / inv) if inv else math.inf
        total_blocks = full + (1 if r.input_len % bt else 0)
        y = (total_blocks) / max(self.prefill_time(r.input_len), 1e-9)       # avg recompute rate, blocks/s
        reassigned = 0.0
        if n_restore and self.t.restore_policy != "default_policy":
            ctx = PolicyContext(total_blocks=n_restore + (total_blocks - hit), miss_blocks=total_blocks - hit,
                                storage_throughput=x_eff, recompute_throughput=y, request_rate=max(rate, 1e-9),
                                p95_seconds=slo)
            reassigned = evaluate(self.policy, ctx).reassigned_hit_blocks
        restored = n_restore - reassigned
        # tokens actually recomputed: misses + reassigned + CacheBlend's selective recompute
        hbm_hit_tokens = (res[0]) * bt
        recompute_tokens = max(0, r.input_len - hit * bt) + int(reassigned * bt) + int(self.t.blend_recompute * blend_blocks * bt)
        cached_tokens = r.input_len - recompute_tokens
        t_recompute = self.prefill_time(recompute_tokens, cached_tokens)
        bytes_restored = restored * self.block_bytes * scale
        t_restore = (restored / x_eff * scale if restored and x_eff != math.inf else 0.0) + bytes_restored / 1e9 * self.t.dequant_s_per_GB
        service = max(t_recompute, t_restore)
        # cache the prompt (prefix keys) and its chunks
        if self.t.prefix_cache:
            self.cache.insert_hbm(r.blocks[:full])
            for i in range(1, len(self.cache.tiers)):
                for kb in r.blocks[:prefix_hit]:
                    if kb in self.cache.tiers[i]:
                        self.cache.touch(kb, i)
        if self.t.nonprefix and r.chunk_ids:
            keys, off, prev = [], 0, None
            for j in range(full):
                c = r.chunk_ids[j]
                off = off + 1 if c == prev else 0
                prev = c
                if c != -1:
                    keys.append(hash((c, off)))
            self.cache.insert_hbm(keys)
        return {
            "service": service, "t_recompute": t_recompute, "t_restore": t_restore,
            "hit_hbm": hits_by_tier[0] * bt, "hit_dram": hits_by_tier[1] * bt, "hit_disk": hits_by_tier[2] * bt,
            "hit_blend": blend_blocks * bt, "recomputed": recompute_tokens, "input": r.input_len,
            "bytes_restored": bytes_restored, "reassigned": reassigned * bt,
        }


@dataclass
class RunConfig:
    load: float = 0.7             # offered load as a fraction of the no-cache prefill capacity
    ttft_slo: float = 2.0         # seconds; also the p95 target handed to KVC-TierSim's policy
    max_requests: int = 2000
    think_time: float = 5.0       # mean seconds between turns of a closed-loop session
    seed: int = 0
    use_trace_time: bool = False  # traces: keep relative timestamps, rescaled to the target load


def nocache_capacity(model, hw, reqs):
    """Sustainable request rate without any KV reuse: min(prefill, decode-memory) bound.

    Load is expressed relative to this so that a 7B and a 70B model, or a chat
    and a 128K-context workload, are compared at the same *utilization* rather
    than the same absolute request rate."""
    e = Engine(model, hw, TECHNIQUES["no_cache"])
    rs = fit_context(reqs[:500], model)
    prefill = 1.0 / max(np.mean([e.prefill_time(r.input_len) for r in rs]), 1e-9)
    tpot_full = e.decode_iter_time(e.kv_hbm_bytes)
    occ = np.mean([e.kv_blocks_live(r) * r.output_len * tpot_full for r in rs])   # block-seconds per request
    decode = e.hbm_total_blocks / max(occ, 1e-9)
    return min(prefill, decode)


def fit_context(reqs: list[Request], model: ModelArch) -> list[Request]:
    """Truncate prompts to the model's context window (keep the head, as a
    prefix cache would see it; LongBench-style middle truncation would only
    change the non-reusable tail). Output is capped to the remaining window."""
    out = []
    for r in reqs:
        lim = model.ctx
        if r.input_len + r.output_len <= lim:
            out.append(r)
            continue
        o = min(r.output_len, max(1, lim // 8))
        n_in = max(r.block_tokens, lim - o)
        nb = -(-n_in // r.block_tokens)
        out.append(Request(rid=r.rid, session=r.session, turn=r.turn, input_len=min(n_in, r.input_len), output_len=o,
                           blocks=r.blocks[:nb], chunk_ids=r.chunk_ids[:nb], arrival=r.arrival,
                           block_tokens=r.block_tokens, out_blocks=[] if n_in < r.input_len else r.out_blocks))
    return out


def replay(reqs: list[Request], model: ModelArch, hw: Hardware, tech: Technique, rc: RunConfig, capacity=None):
    rng = random.Random(rc.seed)
    reqs = fit_context(reqs[: rc.max_requests], model)
    eng = Engine(model, hw, tech, block_tokens=16)
    cap = capacity or nocache_capacity(model, hw, reqs)
    lam = rc.load * cap
    # --- arrivals: open loop for first turns (Poisson or rescaled trace time); later turns follow completion
    by_session: dict[str, list[Request]] = {}
    for r in reqs:
        by_session.setdefault(r.session, []).append(r)
    closed = {s for s, rs in by_session.items() if len(rs) > 1 and any(x.out_blocks for x in rs)}
    has_ts = rc.use_trace_time and all(r.arrival is not None for r in reqs)
    if has_ts:
        span = max(1e-9, reqs[-1].arrival - reqs[0].arrival)
        scale = (len(reqs) / span) / lam
        arr = [(r.arrival - reqs[0].arrival) / scale for r in reqs]
        closed = set()
    else:
        arr, t = [], 0.0
        for r in reqs:
            if r.session in closed and r.turn > 1:
                arr.append(None)
            else:
                t += rng.expovariate(lam)
                arr.append(t)
    next_turn = {}
    for s in closed:
        rs = sorted(by_session[s], key=lambda x: x.turn)
        for a, b_ in zip(rs, rs[1:]):
            next_turn[a.rid] = b_
    events = []   # (time, seq, kind, payload)
    seq = 0
    for r, a in zip(reqs, arr):
        if a is not None:
            heapq.heappush(events, (a, seq, "arrive", r)); seq += 1
    queue: list[tuple[float, Request]] = []
    server_free = 0.0
    live_blocks, live_bytes = 0, 0.0
    waiting_mem: list[tuple[float, Request]] = []
    out = []
    now = 0.0
    seen = 0

    def try_start(now):
        nonlocal server_free, live_blocks, live_bytes, seq
        while queue and server_free <= now:
            arr_t, r = queue[0]
            need = eng.kv_blocks_live(r)
            if live_blocks + need > eng.hbm_total_blocks and live_blocks > 0:
                return  # wait for a decode to finish (memory admission)
            queue.pop(0)
            p = eng.serve_prefill(r, lam, rc.ttft_slo)
            start = max(now, server_free)
            fin = start + p["service"]
            server_free = fin
            live_blocks += need
            live_bytes += need * eng.live_block_bytes
            eng.cache.set_pinned(int(live_blocks * eng.live_block_bytes / eng.block_bytes))
            tpot = eng.decode_iter_time(live_bytes)
            done = fin + r.output_len * tpot
            out.append({"rid": r.rid, "arrival": arr_t, "ttft": fin - arr_t, "queue": start - arr_t, "tpot": tpot,
                        "e2e": done - arr_t, **p})
            heapq.heappush(events, (fin, seq, "prefill_done", r)); seq += 1
            heapq.heappush(events, (done, seq, "decode_done", (r, need))); seq += 1

    while events:
        now, _, kind, payload = heapq.heappop(events)
        if kind == "arrive":
            queue.append((now, payload))
        elif kind == "decode_done":
            r, need = payload
            live_blocks -= need
            live_bytes -= need * eng.live_block_bytes
            eng.cache.set_pinned(int(max(0, live_blocks) * eng.live_block_bytes / eng.block_bytes))
            if tech.prefix_cache and r.out_blocks:
                eng.cache.insert_hbm(r.out_blocks)
            nt = next_turn.get(r.rid)
            if nt is not None:
                heapq.heappush(events, (now + rng.expovariate(1.0 / rc.think_time), seq, "arrive", nt)); seq += 1
        try_start(now)
        if queue and server_free > now:
            heapq.heappush(events, (server_free, seq, "tick", None)); seq += 1
    return summarize(out, eng, tech, model, hw, rc, lam, cap)


def summarize(out, eng, tech, model, hw, rc, lam, cap):
    if not out:
        return {}
    a = {k: np.array([o[k] for o in out], dtype=float) for k in out[0] if k != "rid"}
    tot_in = a["input"].sum()
    dur = max(1e-9, max(o["arrival"] + o["e2e"] for o in out) - min(o["arrival"] for o in out))
    return {
        "technique": tech.name, "model": model.key, "gpu": hw.gpu, "gpu_count": eng.n_gpu,
        "requests": len(out), "offered_rate": lam, "nocache_capacity": cap,
        "ttft_mean": a["ttft"].mean(), "ttft_p50": float(np.percentile(a["ttft"], 50)), "ttft_p99": float(np.percentile(a["ttft"], 99)),
        "service_mean": a["service"].mean(), "queue_mean": a["queue"].mean(),
        "tpot_mean": a["tpot"].mean(), "e2e_mean": a["e2e"].mean(),
        "slo_attainment": float((a["ttft"] <= rc.ttft_slo).mean()),
        "throughput_rps": len(out) / dur,
        "recompute_frac": a["recomputed"].sum() / tot_in,
        "hit_hbm_frac": a["hit_hbm"].sum() / tot_in, "hit_dram_frac": a["hit_dram"].sum() / tot_in,
        "hit_disk_frac": a["hit_disk"].sum() / tot_in, "hit_blend_frac": a["hit_blend"].sum() / tot_in,
        "restored_GB": a["bytes_restored"].sum() / 1e9,
        "ssd_written_GB": eng.cache.writes.get("disk", 0) * eng.block_bytes / 1e9,
        "dram_written_GB": eng.cache.writes.get("dram", 0) * eng.block_bytes / 1e9,
        "block_bytes": eng.block_bytes, "hbm_cache_GB": eng.kv_hbm_bytes / 1e9,
    }
