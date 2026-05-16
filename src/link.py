from dataclasses import dataclass

@dataclass(frozen=True)
class Link:
    name: str
    bandwidth: int
    cost: int
