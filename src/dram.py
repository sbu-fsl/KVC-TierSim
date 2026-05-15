from dataclasses import dataclass

@dataclass(frozen=True)
class DRAM:
    name: str
    bandwidth: int
    capacity: int
