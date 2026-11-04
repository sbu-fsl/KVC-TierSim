"""Hardware pools, loaded from the ``hardware/*.yaml`` catalogs."""

from .loader import load_catalog

_CATALOG = load_catalog()

GPUS = _CATALOG.gpus
DRAMS = _CATALOG.drams
DISKS = _CATALOG.disks
LINKS = _CATALOG.links
MODELS = _CATALOG.models
