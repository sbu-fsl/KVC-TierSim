"""Placement-aware KV-cache simulation engine.

This module extends the original single-tier policy model (see
:mod:`simulator`) with explicit control over where cache-hit blocks live across
the memory hierarchy: GPU HBM (VRAM), CPU DRAM, and Disk.

Model
-----
Every cache-hit block resides in exactly one tier. Restoring a block means
reading it from its tier and (for DRAM/Disk) moving it over the link to the GPU.
Each tier therefore has its own restore throughput (blocks/s):

    X_vram = hbm_bandwidth / B                    (already on the GPU)
    X_dram = 1 / (B/dram_bw + B/link_bw)          (DRAM read, then link)
    X_disk = 1 / (B/disk_bw + B/link_bw)          (disk read, then link)

where ``B`` is bytes per block. Given a residency split (n_v, n_d, n_k) of the
hit blocks, the time to restore all of them is modeled as serial across tiers
(the link is a shared resource), yielding a single blended throughput:

    X_eff = (n_v + n_d + n_k) / (n_v/X_vram + n_d/X_dram + n_k/X_disk)

``X_eff`` is then fed into the existing policy engine (default / all-compute /
performance-aware), so pushing residency toward VRAM/DRAM raises the restore
rate and shifts every policy's latency and SLO outcome. Recompute throughput
``Y`` is unchanged from the original model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from simulator import (
    HardwareConfig,
    evaluate_all_compute,
    evaluate_default_policy,
    evaluate_performance_aware,
)
from src.pool import DISKS, DRAMS, GPUS, LINKS

TIER_ORDER = ("vram", "dram", "disk")


@dataclass(frozen=True)
class Workload:
    """A single simulation point."""

    total_blocks: int
    miss_blocks: int
    request_rate: float
    p95_seconds: float
    model_params: float = 8e9
    tokens_per_block: int = 16
    bytes_per_block: float = 2e6
    gpu_eta: float = 0.5

    @property
    def hit_blocks(self) -> int:
        return max(0, self.total_blocks - self.miss_blocks)


@dataclass(frozen=True)
class Placement:
    """Residency of cache-hit blocks across tiers.

    ``mode='auto'`` fills VRAM, then DRAM, then Disk by real capacity.
    ``mode='manual'`` uses ``fractions`` (a mapping over TIER_ORDER, normalized).
    """

    mode: str = "auto"
    fractions: dict[str, float] = field(
        default_factory=lambda: {"vram": 0.0, "dram": 0.0, "disk": 1.0}
    )


@dataclass(frozen=True)
class TierState:
    tier: str
    label: str
    capacity_blocks: float
    resident_blocks: float
    overflow_blocks: float
    bandwidth_bytes_per_s: float
    restore_throughput_blocks_per_s: float
    utilization: float  # resident / capacity, clamped display value


def _tier_capacities_blocks(hardware: HardwareConfig, workload: Workload) -> dict[str, float]:
    """Usable KV-cache capacity of each tier, in blocks."""
    gpu = GPUS[hardware.gpu_key]
    dram = DRAMS[hardware.dram_key]
    disk = DISKS[hardware.disk_key]

    model_size_bytes = workload.model_params * 2  # bf16: 2 bytes / param
    vram_bytes = gpu.hbm_capacity * hardware.gpu_count - model_size_bytes
    dram_bytes = dram.capacity * hardware.dram_count
    disk_bytes = disk.capacity

    b = workload.bytes_per_block
    return {
        "vram": max(0.0, vram_bytes) / b,
        "dram": max(0.0, dram_bytes) / b,
        "disk": max(0.0, disk_bytes) / b,
    }


def _tier_throughputs(hardware: HardwareConfig, workload: Workload) -> dict[str, float]:
    """Per-tier restore throughput in blocks/s."""
    gpu = GPUS[hardware.gpu_key]
    dram = DRAMS[hardware.dram_key]
    disk = DISKS[hardware.disk_key]
    link = LINKS[hardware.link_key]
    b = workload.bytes_per_block

    x_vram = gpu.hbm_bandwidth / b
    x_dram = 1.0 / (b / dram.bandwidth + b / link.bandwidth)
    x_disk = 1.0 / (b / disk.bandwidth + b / link.bandwidth)
    return {"vram": x_vram, "dram": x_dram, "disk": x_disk}


def _tier_bandwidths(hardware: HardwareConfig) -> dict[str, float]:
    return {
        "vram": GPUS[hardware.gpu_key].hbm_bandwidth,
        "dram": DRAMS[hardware.dram_key].bandwidth,
        "disk": DISKS[hardware.disk_key].bandwidth,
    }


def _resolve_residency(
    placement: Placement,
    hit_blocks: int,
    capacities: dict[str, float],
) -> dict[str, float]:
    """Return blocks resident in each tier for the given placement.

    Auto mode waterfalls by capacity. Manual mode distributes by normalized
    fractions. In both cases blocks that do not fit anywhere spill to disk as
    overflow (reported separately), so the residency counts always sum to
    ``hit_blocks``.
    """
    if hit_blocks <= 0:
        return {tier: 0.0 for tier in TIER_ORDER}

    resident: dict[str, float] = {tier: 0.0 for tier in TIER_ORDER}

    if placement.mode == "auto":
        remaining = float(hit_blocks)
        for tier in TIER_ORDER:
            if remaining <= 0:
                break
            take = min(remaining, capacities[tier]) if tier != "disk" else remaining
            resident[tier] = take
            remaining -= take
        # Anything still remaining did not fit even on disk; park it on disk.
        if remaining > 0:
            resident["disk"] += remaining
        return resident

    # Manual: normalize fractions over the tier order.
    raw = {tier: max(0.0, float(placement.fractions.get(tier, 0.0))) for tier in TIER_ORDER}
    total = sum(raw.values())
    if total <= 0:
        raw = {"vram": 0.0, "dram": 0.0, "disk": 1.0}
        total = 1.0
    for tier in TIER_ORDER:
        resident[tier] = hit_blocks * raw[tier] / total
    return resident


def _effective_storage_throughput(
    residency: dict[str, float],
    throughputs: dict[str, float],
) -> float:
    """Blended restore throughput (blocks/s) over the resident hit blocks."""
    total = sum(residency.values())
    if total <= 0:
        return throughputs["disk"]  # nominal; no restore work anyway

    inv_time = 0.0
    for tier in TIER_ORDER:
        n = residency[tier]
        if n <= 0:
            continue
        x = throughputs[tier]
        if x <= 0:
            return 0.0
        inv_time += n / x
    if inv_time <= 0:
        return math.inf
    return total / inv_time


def _recompute_throughput(hardware: HardwareConfig, workload: Workload) -> float:
    gpu = GPUS[hardware.gpu_key]
    t_per_token = gpu.gpu_compute_band(
        workload.model_params, gpu_count=hardware.gpu_count, eta=workload.gpu_eta
    )
    if t_per_token <= 0:
        return math.inf
    return 1.0 / (workload.tokens_per_block * t_per_token)


def _build_tier_states(
    hardware: HardwareConfig,
    residency: dict[str, float],
    capacities: dict[str, float],
    throughputs: dict[str, float],
) -> list[TierState]:
    bandwidths = _tier_bandwidths(hardware)
    labels = {
        "vram": f"GPU HBM ({GPUS[hardware.gpu_key].name})",
        "dram": f"CPU DRAM ({DRAMS[hardware.dram_key].name} x{hardware.dram_count})",
        "disk": f"Disk ({DISKS[hardware.disk_key].name})",
    }
    states: list[TierState] = []
    for tier in TIER_ORDER:
        cap = capacities[tier]
        resident = residency[tier]
        overflow = max(0.0, resident - cap)
        utilization = (resident / cap) if cap > 0 else (math.inf if resident > 0 else 0.0)
        states.append(
            TierState(
                tier=tier,
                label=labels[tier],
                capacity_blocks=cap,
                resident_blocks=resident,
                overflow_blocks=overflow,
                bandwidth_bytes_per_s=bandwidths[tier],
                restore_throughput_blocks_per_s=throughputs[tier],
                utilization=utilization,
            )
        )
    return states


def _speedup(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.inf


def simulate(
    hardware: HardwareConfig,
    workload: Workload,
    placement: Placement,
) -> dict[str, Any]:
    """Run all three policies for one hardware / workload / placement point."""
    if workload.total_blocks <= 0:
        raise ValueError("total_blocks must be greater than zero")
    if workload.miss_blocks < 0 or workload.miss_blocks > workload.total_blocks:
        raise ValueError("miss_blocks must be in [0, total_blocks]")
    if workload.request_rate <= 0:
        raise ValueError("request_rate must be greater than zero")
    if workload.p95_seconds <= 0:
        raise ValueError("p95_seconds must be greater than zero")

    capacities = _tier_capacities_blocks(hardware, workload)
    throughputs = _tier_throughputs(hardware, workload)
    residency = _resolve_residency(placement, workload.hit_blocks, capacities)
    tier_states = _build_tier_states(hardware, residency, capacities, throughputs)

    storage_throughput = _effective_storage_throughput(residency, throughputs)
    recompute_throughput = _recompute_throughput(hardware, workload)

    args = (
        hardware,
        workload.total_blocks,
        workload.miss_blocks,
        storage_throughput,
        recompute_throughput,
        workload.request_rate,
        workload.p95_seconds,
    )
    performance_aware = evaluate_performance_aware(*args)
    default_policy = evaluate_default_policy(*args)
    all_compute = evaluate_all_compute(*args)

    fits = all(state.overflow_blocks == 0.0 for state in tier_states)

    return {
        "hardware": asdict(hardware),
        "workload": asdict(workload),
        "placement": {"mode": placement.mode, "fractions": placement.fractions},
        "capacity_fits": fits,
        "storage_throughput": storage_throughput,
        "recompute_throughput": recompute_throughput,
        "tiers": [asdict(state) for state in tier_states],
        "performance_aware": asdict(performance_aware),
        "default_policy": asdict(default_policy),
        "all_compute": asdict(all_compute),
        "speedup_performance_aware_vs_default": _speedup(
            default_policy.total_time, performance_aware.total_time
        ),
        "speedup_performance_aware_vs_all_compute": _speedup(
            all_compute.total_time, performance_aware.total_time
        ),
    }
