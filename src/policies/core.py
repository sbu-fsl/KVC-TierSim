"""Shared policy math — the hardware-derived model every policy runs on.

A *policy* only decides one thing: how many cache-hit blocks ``k`` to reassign
from restoration (read from a storage tier) to recomputation (on the GPU).
Everything else — restore/recompute throughputs, block allocation, latency, and
the SLO constraint checks — is identical across policies and lives here.

Flow:  hardware layer computes X (restore blocks/s) and Y (recompute blocks/s)
       -> PolicyContext bundles them with the workload (N, M, r, p)
       -> a Policy.decide(ctx) returns a PolicyDecision(k, mode)
       -> evaluate(policy, ctx) applies the shared math -> PolicyResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyResult:
    """Full outcome for one policy on one workload point (JSON-serializable)."""

    policy: str
    stack_key: str
    total_blocks: int
    miss_blocks: int
    hit_blocks: int
    reassigned_hit_blocks: float
    request_rate: float
    p95_seconds: float
    slo_seconds: float
    meets_slo: bool
    success_rate: float
    slo_margin: float
    decision_mode: str
    latency_ratio: float
    gpu_utilization: float
    storage_utilization: float
    latency_violation: float
    gpu_violation: float
    storage_violation: float
    max_violation: float
    storage_blocks: float
    recompute_blocks: float
    storage_throughput: float
    recompute_throughput: float
    storage_time: float
    recompute_time: float
    total_time: float


@dataclass(frozen=True)
class PolicyDecision:
    """What a policy returns: how many hit blocks to recompute, and a label."""

    reassigned_hit_blocks: float
    decision_mode: str


@dataclass(frozen=True)
class PolicyContext:
    """Everything a policy needs, all derived from shared hardware math.

    ``storage_throughput`` (X) and ``recompute_throughput`` (Y) are supplied by
    the caller (CLI single-tier model or placement-aware engine); a policy never
    recomputes them. The helper methods let a policy score candidate ``k`` values
    without duplicating the allocation/constraint math.
    """

    total_blocks: int
    miss_blocks: int
    storage_throughput: float
    recompute_throughput: float
    request_rate: float
    p95_seconds: float
    stack_key: str = "custom"

    @property
    def hit_blocks(self) -> int:
        return max(0, self.total_blocks - self.miss_blocks)

    def clamp(self, reassigned_hit_blocks: float) -> float:
        """Clamp a proposed k into the feasible range [0, hit_blocks]."""
        return max(0.0, min(float(self.hit_blocks), float(reassigned_hit_blocks)))

    def allocation(self, reassigned_hit_blocks: float) -> tuple[float, float, float, float]:
        """Return (storage_blocks, recompute_blocks, storage_time, recompute_time)."""
        k = self.clamp(reassigned_hit_blocks)
        recompute_blocks = float(self.miss_blocks) + k
        storage_blocks = float(self.hit_blocks) - k
        storage_time = (
            storage_blocks / self.storage_throughput
            if self.storage_throughput > 0
            else math.inf
        )
        recompute_time = (
            recompute_blocks / self.recompute_throughput
            if self.recompute_throughput > 0
            else math.inf
        )
        return storage_blocks, recompute_blocks, storage_time, recompute_time

    def max_violation(self, reassigned_hit_blocks: float) -> float:
        """Worst SLO-constraint violation for a candidate k (0.0 means it passes)."""
        _, _, storage_time, recompute_time = self.allocation(reassigned_hit_blocks)
        total_time = max(storage_time, recompute_time)
        if self.p95_seconds <= 0:
            return math.inf
        latency_violation = max(0.0, (total_time / self.p95_seconds) - 1.0)
        gpu_violation = max(0.0, self.request_rate * recompute_time - 1.0)
        storage_violation = max(0.0, self.request_rate * storage_time - 1.0)
        return max(latency_violation, gpu_violation, storage_violation)


def evaluate(policy: "PolicyBase", ctx: PolicyContext) -> PolicyResult:
    """Run a policy on a context and apply the shared hardware/SLO math."""
    if ctx.request_rate <= 0:
        raise ValueError("request_rate must be greater than zero")
    if ctx.p95_seconds <= 0:
        raise ValueError("p95_seconds must be greater than zero")

    decision = policy.decide(ctx)
    k = ctx.clamp(decision.reassigned_hit_blocks)
    storage_blocks, recompute_blocks, storage_time, recompute_time = ctx.allocation(k)
    total_time = max(storage_time, recompute_time)

    latency_ratio = total_time / ctx.p95_seconds
    gpu_utilization = ctx.request_rate * recompute_time
    storage_utilization = ctx.request_rate * storage_time
    latency_violation = max(0.0, latency_ratio - 1.0)
    gpu_violation = max(0.0, gpu_utilization - 1.0)
    storage_violation = max(0.0, storage_utilization - 1.0)
    max_violation = max(latency_violation, gpu_violation, storage_violation)
    meets_slo = max_violation == 0.0

    return PolicyResult(
        policy=policy.tag,
        stack_key=ctx.stack_key,
        total_blocks=ctx.total_blocks,
        miss_blocks=ctx.miss_blocks,
        hit_blocks=ctx.hit_blocks,
        reassigned_hit_blocks=k,
        request_rate=ctx.request_rate,
        p95_seconds=ctx.p95_seconds,
        slo_seconds=ctx.p95_seconds,
        meets_slo=meets_slo,
        success_rate=1.0 if meets_slo else 0.0,
        slo_margin=ctx.p95_seconds - total_time,
        decision_mode=decision.decision_mode,
        latency_ratio=latency_ratio,
        gpu_utilization=gpu_utilization,
        storage_utilization=storage_utilization,
        latency_violation=latency_violation,
        gpu_violation=gpu_violation,
        storage_violation=storage_violation,
        max_violation=max_violation,
        storage_blocks=storage_blocks,
        recompute_blocks=recompute_blocks,
        storage_throughput=ctx.storage_throughput,
        recompute_throughput=ctx.recompute_throughput,
        storage_time=storage_time,
        recompute_time=recompute_time,
        total_time=total_time,
    )


# Imported lazily for type hints only; avoids a circular import at module load.
if False:  # pragma: no cover
    from .base import PolicyBase
