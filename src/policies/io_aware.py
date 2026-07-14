"""IO-aware policy.

Balances the restore and recompute paths so they finish together:

    k = N*Y / (X + Y) - M      (clamped to [0, hit_blocks])

where N is total blocks, M misses, X the restore rate, and Y the recompute rate
(in blocks/s). When that balanced split still violates the SLO, fall back to a
best-effort search over the handful of k values where an individual constraint
becomes tight, picking the one with the lowest worst-case violation.
"""

from __future__ import annotations

import math

from .base import PolicyBase
from .core import PolicyContext, PolicyDecision


class IOAwarePolicy(PolicyBase):
    name = "performance_aware"
    label = "IO-aware policy"
    description = (
        "Balance restore and recompute so both paths finish together; "
        "fall back to a best-effort split under SLO pressure."
    )
    color = "#4f8cff"
    order = 10

    def _balanced_k(self, ctx: PolicyContext) -> float:
        hit = ctx.hit_blocks
        if hit <= 0:
            return 0.0
        y = ctx.recompute_throughput
        x = ctx.storage_throughput
        if y <= 0:
            return 0.0
        if x <= 0:
            return float(hit)
        k = (ctx.total_blocks * y) / (y + x) - ctx.miss_blocks
        return ctx.clamp(k)

    def _best_effort_k(self, ctx: PolicyContext) -> float:
        hit = ctx.hit_blocks
        if hit <= 0:
            return 0.0

        x = ctx.storage_throughput
        y = ctx.recompute_throughput
        r = ctx.request_rate
        p = ctx.p95_seconds
        miss = ctx.miss_blocks

        # Candidate k values: the endpoints, the balanced point, and the points
        # where each individual constraint (rate / latency) becomes tight.
        candidates = {0.0, float(hit), self._balanced_k(ctx)}
        if y > 0:
            candidates.add((y / r) - miss)
            candidates.add(p * y - miss)
        if x > 0:
            candidates.add(ctx.total_blocks - miss - (x / r))
            candidates.add(ctx.total_blocks - miss - (p * x))

        best_k = 0.0
        best_score = math.inf
        best_total_time = math.inf
        for candidate in candidates:
            k = ctx.clamp(candidate)
            _, _, storage_time, recompute_time = ctx.allocation(k)
            score = ctx.max_violation(k)
            total_time = max(storage_time, recompute_time)
            if score < best_score or (
                score == best_score and total_time < best_total_time
            ):
                best_score = score
                best_total_time = total_time
                best_k = k
        return best_k

    def decide(self, ctx: PolicyContext) -> PolicyDecision:
        balanced = self._balanced_k(ctx)
        if ctx.max_violation(balanced) == 0.0:
            return PolicyDecision(
                reassigned_hit_blocks=balanced, decision_mode="balanced"
            )
        return PolicyDecision(
            reassigned_hit_blocks=self._best_effort_k(ctx),
            decision_mode="best_effort",
        )


POLICY = IOAwarePolicy()
