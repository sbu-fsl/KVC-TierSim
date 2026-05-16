from dataclasses import dataclass

@dataclass(frozen=True)
class Disk:
    name: str
    bandwidth: int
    capacity: int
    cost: int
