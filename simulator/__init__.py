"""KVC-TierSim: restore-vs-recompute simulator for tiered KV-cache storage.

Layout:
    simulator.core        hardware catalog loader, engine, and pluggable policies
    simulator.experiment  single-point experiment runner (``python -m simulator``)
    simulator.sweep       P95 x request-rate sweep + figures
    simulator.overhead    scheduler-overhead sweep
    simulator.figures     publication figures
    simulator.api         FastAPI backend serving simulator/frontend
"""

from simulator.experiment import (
    STACK_PRESETS,
    ExperimentConfig,
    HardwareConfig,
    hardware_throughputs,
    run_experiments,
)

__all__ = [
    "STACK_PRESETS",
    "ExperimentConfig",
    "HardwareConfig",
    "hardware_throughputs",
    "run_experiments",
]
