"""Policy registry."""

from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path

from .base import PolicyBase
from .core import PolicyContext, PolicyDecision, PolicyResult, evaluate

# Infrastructure modules that never contain a policy.
_EXCLUDE = {"base", "core"}


def _discover() -> dict[str, PolicyBase]:
    registry: dict[str, PolicyBase] = {}
    package_dir = Path(__file__).parent
    for module_info in iter_modules([str(package_dir)]):
        name = module_info.name
        if name in _EXCLUDE or name.startswith("_"):
            continue
        module = import_module(f"{__name__}.{name}")
        policy = getattr(module, "POLICY", None)
        if isinstance(policy, PolicyBase):
            if policy.name in registry:
                raise ValueError(f"Duplicate policy name: {policy.name}")
            registry[policy.name] = policy
    return registry


_REGISTRY = _discover()


def get_policies() -> list[PolicyBase]:
    """All registered policies, ordered by ``order`` then name."""
    return sorted(_REGISTRY.values(), key=lambda p: (p.order, p.name))


def get_policy(name: str) -> PolicyBase:
    return _REGISTRY[name]


def policy_keys() -> tuple[str, ...]:
    return tuple(policy.name for policy in get_policies())


def policy_meta() -> list[dict]:
    return [policy.meta() for policy in get_policies()]


__all__ = [
    "PolicyBase",
    "PolicyContext",
    "PolicyDecision",
    "PolicyResult",
    "evaluate",
    "get_policies",
    "get_policy",
    "policy_keys",
    "policy_meta",
]
