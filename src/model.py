from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    """An LLM workload preset describing model size and KV-block granularity."""

    name: str
    params: float
    tokens_per_block: int = 16
    bytes_per_block: float = 2e6
    gpu_eta: float = 0.5
