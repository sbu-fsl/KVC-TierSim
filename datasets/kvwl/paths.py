"""Filesystem layout of the datasets/ project, in one place.

    datasets/
      catalog/   models.yaml, workloads.yaml
      survey/    corpus mining + ranking (survey/out/ holds the top-50 lists)
      data/      results/, analysis/, advisor/, workload_profiles.csv, workload_model_matrix.csv

Big data (raw sources, request streams, download cache) lives in the data store:
$KVWL_DATA_DIR when set (e.g. /mnt/gpfs/kvc_tiersim_kvwl on ibm1.fsl), else datasets/data/:
      <store>/raw/  <store>/streams/  <store>/.cache/
      figures/   generated figures and the interactive explorer
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # datasets/
REPO_ROOT = os.path.dirname(ROOT)                                      # repository root

DATA = os.path.join(ROOT, "data")
DATA_STORE = os.path.abspath(os.path.expanduser(os.environ["KVWL_DATA_DIR"])) if os.environ.get("KVWL_DATA_DIR") else DATA
RAW = os.path.join(DATA_STORE, "raw")
STREAMS = os.path.join(DATA_STORE, "streams")
DOWNLOAD_CACHE = os.path.join(DATA_STORE, ".cache")
RESULTS = os.path.join(DATA, "results")
ANALYSIS = os.path.join(DATA, "analysis")
ADVISOR = os.path.join(DATA, "advisor")
CATALOG = os.path.join(ROOT, "catalog")
SURVEY = os.path.join(ROOT, "survey")
SURVEY_OUT = os.path.join(SURVEY, "out")
FIGURES = os.path.join(ROOT, "figures")


def ensure_simulator_importable() -> None:
    """Make the sibling ``simulator`` package importable without ``pip install -e .``."""
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
