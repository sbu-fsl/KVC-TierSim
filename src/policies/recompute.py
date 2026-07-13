"""All-compute policy: recompute every block on the GPU (k = all hits)."""

from __future__ import annotations

from .base import PolicyBase
from .core import PolicyContext, PolicyDecision


class RecomputePolicy(PolicyBase):
    name = "all_compute"
    label = "Recompute"
    description = "Recompute every block on the GPU; never restore from storage."
    color = "#33bf59"
    order = 30

    def decide(self, ctx: PolicyContext) -> PolicyDecision:
        return PolicyDecision(
            reassigned_hit_blocks=float(ctx.hit_blocks),
            decision_mode="fixed_all_compute",
        )


POLICY = RecomputePolicy()
