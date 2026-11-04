"""Default restore policy: restore every hit, recompute only misses (k = 0)."""

from __future__ import annotations

from .base import PolicyBase
from .core import PolicyContext, PolicyDecision


class RestorePolicy(PolicyBase):
    name = "default_policy"
    label = "Restore"
    description = "Restore every cache-hit block from storage; recompute only misses."
    color = "#e01c1c"
    order = 20
    policy_tag = "restore_hits_recompute_misses"

    def decide(self, ctx: PolicyContext) -> PolicyDecision:
        return PolicyDecision(reassigned_hit_blocks=0.0, decision_mode="fixed_default")


POLICY = RestorePolicy()
