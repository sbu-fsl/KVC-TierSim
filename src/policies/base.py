"""Base class for KV-cache placement policies."""

from __future__ import annotations

from .core import PolicyContext, PolicyDecision


class PolicyBase:
    # Stable identifier used as the JSON key / registry key (snake_case).
    name: str = "policy"
    # Human-facing label shown in the dashboard and plots.
    label: str = "Policy"
    # One-line explanation of the strategy.
    description: str = ""
    # Hex color used consistently across the UI.
    color: str = "#8a95ad"
    # Display order (ascending); lower shows first.
    order: int = 100
    # Value stored in PolicyResult.policy. Defaults to ``name`` when empty.
    policy_tag: str = ""

    @property
    def tag(self) -> str:
        return self.policy_tag or self.name

    def decide(self, ctx: PolicyContext) -> PolicyDecision:
        """Return how many cache-hit blocks to reassign to recomputation."""
        raise NotImplementedError

    def meta(self) -> dict:
        """Metadata for the API / dashboard."""
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "color": self.color,
            "order": self.order,
        }
