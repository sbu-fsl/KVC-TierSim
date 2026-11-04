"""Typed hardware and model definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    hbm_bandwidth: float
    hbm_capacity: float
    peak_flops: float
    cost: float
    # Ceiling on the interconnect that feeds this GPU (its NVLink generation, or
    # PCIe for parts without one). A catalog link faster than this cannot be
    # attached to the part; 0.0 leaves it uncapped.
    max_link_bandwidth: float = 0.0

    def effective_link_bandwidth(self, link_bandwidth: float) -> float:
        if self.max_link_bandwidth <= 0.0:
            return link_bandwidth
        return min(link_bandwidth, self.max_link_bandwidth)

    def gpu_compute_band(
        self, model_params: float, gpu_count: int = 1, eta: float = 0.6
    ) -> float:
        """Seconds of GPU compute per generated token (decode-bound estimate)."""
        flops_per_token = 2 * model_params
        total_flops = self.peak_flops * gpu_count * eta
        return flops_per_token / total_flops


@dataclass(frozen=True)
class DRAM:
    name: str
    bandwidth: float
    capacity: float
    cost: float


@dataclass(frozen=True)
class Disk:
    name: str
    bandwidth: float
    capacity: float
    cost: float


@dataclass(frozen=True)
class Link:
    name: str
    bandwidth: float
    cost: float


@dataclass(frozen=True)
class ModelPreset:
    """An LLM workload preset describing model size and KV-block granularity."""

    name: str
    params: float
    tokens_per_block: int = 16
    bytes_per_block: float = 2e6
    gpu_eta: float = 0.5
