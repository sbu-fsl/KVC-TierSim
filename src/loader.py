"""Load hardware and model catalogs from the ``hardware/`` YAML files.

Each YAML file maps a stable string key to a component's fields. The loader
turns those into frozen dataclass instances so the rest of the simulator works
with typed objects rather than dicts. Values may use YAML float syntax such as
``3.36e12`` for bandwidths and capacities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml

from .disk import Disk
from .dram import DRAM
from .gpu import GPU
from .link import Link
from .model import ModelPreset

# hardware/ lives next to the repository root, i.e. one level above src/.
HARDWARE_DIR = Path(__file__).resolve().parent.parent / "hardware"

T = TypeVar("T")


def _load_yaml(filename: str) -> dict[str, dict[str, Any]]:
    path = HARDWARE_DIR / filename
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping of key -> fields")
    return data


def _coerce_numeric(fields: dict[str, Any]) -> dict[str, Any]:
    """Coerce non-name fields to float.

    PyYAML follows the YAML 1.1 core schema, under which an unsigned exponent
    such as ``2e9`` parses as a *string* (a decimal point or signed exponent is
    required for it to be a float). Rather than force ``2.0e9`` everywhere in the
    catalogs, coerce every numeric field here.
    """
    coerced: dict[str, Any] = {}
    for name, value in fields.items():
        if name == "name":
            coerced[name] = value
        else:
            coerced[name] = float(value)
    return coerced


def _build_catalog(filename: str, cls: Type[T]) -> dict[str, T]:
    """Instantiate ``cls`` for every entry in ``filename``."""
    catalog: dict[str, T] = {}
    for key, fields in _load_yaml(filename).items():
        catalog[key] = cls(**_coerce_numeric(fields))
    return catalog


def load_gpus() -> dict[str, GPU]:
    return _build_catalog("gpus.yaml", GPU)


def load_drams() -> dict[str, DRAM]:
    return _build_catalog("dram.yaml", DRAM)


def load_disks() -> dict[str, Disk]:
    return _build_catalog("disks.yaml", Disk)


def load_links() -> dict[str, Link]:
    return _build_catalog("links.yaml", Link)


def load_models() -> dict[str, ModelPreset]:
    return _build_catalog("models.yaml", ModelPreset)


@dataclass(frozen=True)
class Catalog:
    """Bundle of every hardware/model catalog, loaded once."""

    gpus: dict[str, GPU]
    drams: dict[str, DRAM]
    disks: dict[str, Disk]
    links: dict[str, Link]
    models: dict[str, ModelPreset]


def load_catalog() -> Catalog:
    return Catalog(
        gpus=load_gpus(),
        drams=load_drams(),
        disks=load_disks(),
        links=load_links(),
        models=load_models(),
    )
