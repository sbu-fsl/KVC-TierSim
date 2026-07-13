"""Hardware pools, loaded from the ``hardware/*.yaml`` catalogs.

The catalogs used to be hard-coded here. They now live in editable YAML files
under ``hardware/`` and are read into typed objects by :mod:`src.loader`. The
public names (``GPUS``, ``DRAMS``, ``DISKS``, ``LINKS``, ``MODELS``) are kept so
existing imports such as ``from src.pool import GPUS`` keep working.
"""

from .loader import load_catalog

_CATALOG = load_catalog()

GPUS = _CATALOG.gpus
DRAMS = _CATALOG.drams
DISKS = _CATALOG.disks
LINKS = _CATALOG.links
MODELS = _CATALOG.models
