"""FastAPI backend for the KV-cache tier-placement simulator.

Endpoints
---------
GET  /api/catalog   Hardware + model catalogs and stack presets (for the UI).
POST /api/simulate  Run all three policies for one hardware/workload/placement.
POST /api/sweep     Sweep P95 x request-rate into per-policy pass/fail grids,
                    plus a success-rate-vs-cache-ratio curve.
GET  /              Serves the dashboard (frontend/index.html).

Run with:  uvicorn api.app:app --reload
"""

from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from simulator import HardwareConfig, STACK_PRESETS
from src.engine import Placement, Workload, simulate
from src.pool import DISKS, DRAMS, GPUS, LINKS, MODELS

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="KV-Cache Tier Placement Simulator", version="1.0.0")


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class HardwareSpec(BaseModel):
    gpu_key: str = "H200"
    dram_key: str = "DDR5"
    disk_key: str = "NVMe"
    link_key: str = "NVLink"
    gpu_count: int = Field(default=1, ge=1)
    dram_count: int = Field(default=4, ge=1)


class WorkloadSpec(BaseModel):
    total_blocks: int = Field(default=50000, gt=0)
    miss_blocks: int = Field(default=500, ge=0)
    request_rate: float = Field(default=0.02, gt=0)
    p95_seconds: float = Field(default=10.0, gt=0)
    model_params: float = Field(default=8e9, gt=0)
    tokens_per_block: int = Field(default=16, gt=0)
    bytes_per_block: float = Field(default=2e6, gt=0)
    gpu_eta: float = Field(default=0.5, gt=0, le=1)


class PlacementSpec(BaseModel):
    mode: str = "auto"  # "auto" | "manual"
    vram: float = 0.0
    dram: float = 0.0
    disk: float = 1.0


class SimulateRequest(BaseModel):
    hardware: HardwareSpec = HardwareSpec()
    workload: WorkloadSpec = WorkloadSpec()
    placement: PlacementSpec = PlacementSpec()


class SweepRequest(BaseModel):
    hardware: HardwareSpec = HardwareSpec()
    placement: PlacementSpec = PlacementSpec()
    total_blocks: int = Field(default=10000, gt=0)
    hit_ratio: float = Field(default=50.0, ge=0, le=100)
    p95_values: list[float] = Field(
        default_factory=lambda: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    )
    request_rates: list[float] = Field(
        default_factory=lambda: [round(0.01 * i, 2) for i in range(1, 31)]
    )
    cache_ratio_sweep: list[float] = Field(
        default_factory=lambda: [float(v) for v in range(0, 101, 10)]
    )
    # Workload knobs shared across the sweep.
    model_params: float = Field(default=8e9, gt=0)
    tokens_per_block: int = Field(default=16, gt=0)
    bytes_per_block: float = Field(default=2e6, gt=0)
    gpu_eta: float = Field(default=0.5, gt=0, le=1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _hardware_config(spec: HardwareSpec) -> HardwareConfig:
    for key, catalog, kind in (
        (spec.gpu_key, GPUS, "gpu"),
        (spec.dram_key, DRAMS, "dram"),
        (spec.disk_key, DISKS, "disk"),
        (spec.link_key, LINKS, "link"),
    ):
        if key not in catalog:
            raise HTTPException(status_code=400, detail=f"Unknown {kind} key: {key}")
    return HardwareConfig(
        stack_key="custom",
        gpu_key=spec.gpu_key,
        dram_key=spec.dram_key,
        disk_key=spec.disk_key,
        link_key=spec.link_key,
        gpu_count=spec.gpu_count,
        dram_count=spec.dram_count,
    )


def _placement(spec: PlacementSpec) -> Placement:
    return Placement(
        mode=spec.mode,
        fractions={"vram": spec.vram, "dram": spec.dram, "disk": spec.disk},
    )


def _workload(spec: WorkloadSpec) -> Workload:
    return Workload(
        total_blocks=spec.total_blocks,
        miss_blocks=min(spec.miss_blocks, spec.total_blocks),
        request_rate=spec.request_rate,
        p95_seconds=spec.p95_seconds,
        model_params=spec.model_params,
        tokens_per_block=spec.tokens_per_block,
        bytes_per_block=spec.bytes_per_block,
        gpu_eta=spec.gpu_eta,
    )


def _sanitize(value: Any) -> Any:
    """Replace inf/nan with JSON-friendly stand-ins recursively."""
    if isinstance(value, float):
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _policy_pass_state(policy_row: dict[str, Any]) -> dict[str, bool]:
    rpc_pass = policy_row["gpu_violation"] == 0.0 and policy_row["storage_violation"] == 0.0
    latency_pass = policy_row["latency_violation"] == 0.0
    return {
        "rpc_pass": rpc_pass,
        "latency_pass": latency_pass,
        "pass": rpc_pass and latency_pass,
    }


POLICY_KEYS = ("default_policy", "all_compute", "performance_aware")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/catalog")
def catalog() -> JSONResponse:
    def dump(mapping):
        return {key: asdict(value) for key, value in mapping.items()}

    payload = {
        "gpus": dump(GPUS),
        "drams": dump(DRAMS),
        "disks": dump(DISKS),
        "links": dump(LINKS),
        "models": dump(MODELS),
        "stacks": {key: asdict(cfg) for key, cfg in STACK_PRESETS.items()},
    }
    return JSONResponse(_sanitize(payload))


@app.post("/api/simulate")
def simulate_endpoint(request: SimulateRequest) -> JSONResponse:
    hardware = _hardware_config(request.hardware)
    workload = _workload(request.workload)
    placement = _placement(request.placement)
    try:
        result = simulate(hardware, workload, placement)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(_sanitize(result))


@app.post("/api/sweep")
def sweep_endpoint(request: SweepRequest) -> JSONResponse:
    hardware = _hardware_config(request.hardware)
    placement = _placement(request.placement)

    p95_values = request.p95_values
    request_rates = request.request_rates
    if not p95_values or not request_rates:
        raise HTTPException(status_code=400, detail="p95_values and request_rates required")

    def make_workload(total_blocks: int, miss_blocks: int, rate: float, p95: float) -> Workload:
        return Workload(
            total_blocks=total_blocks,
            miss_blocks=miss_blocks,
            request_rate=rate,
            p95_seconds=p95,
            model_params=request.model_params,
            tokens_per_block=request.tokens_per_block,
            bytes_per_block=request.bytes_per_block,
            gpu_eta=request.gpu_eta,
        )

    # Pass/fail grids (rows = p95 values, cols = request rates), one per policy.
    miss_blocks = int(round(request.total_blocks * (100 - request.hit_ratio) / 100))
    grids: dict[str, list[list[dict[str, bool]]]] = {k: [] for k in POLICY_KEYS}
    for p95 in p95_values:
        rows = {k: [] for k in POLICY_KEYS}
        for rate in request_rates:
            wl = make_workload(request.total_blocks, miss_blocks, rate, p95)
            result = simulate(hardware, wl, placement)
            for key in POLICY_KEYS:
                rows[key].append(_policy_pass_state(result[key]))
        for key in POLICY_KEYS:
            grids[key].append(rows[key])

    # Success-rate vs cache-ratio curve.
    total_cases = len(p95_values) * len(request_rates)
    success_rates: dict[str, list[float]] = {k: [] for k in POLICY_KEYS}
    for cache_ratio in request.cache_ratio_sweep:
        miss = int(round(request.total_blocks * (100 - cache_ratio) / 100))
        counts = {k: 0 for k in POLICY_KEYS}
        for p95 in p95_values:
            for rate in request_rates:
                wl = make_workload(request.total_blocks, miss, rate, p95)
                result = simulate(hardware, wl, placement)
                for key in POLICY_KEYS:
                    counts[key] += int(bool(result[key]["meets_slo"]))
        for key in POLICY_KEYS:
            success_rates[key].append(100.0 * counts[key] / total_cases if total_cases else 0.0)

    payload = {
        "hardware": asdict(hardware),
        "total_blocks": request.total_blocks,
        "hit_ratio": request.hit_ratio,
        "miss_blocks": miss_blocks,
        "p95_values": p95_values,
        "request_rates": request_rates,
        "cache_ratio_sweep": request.cache_ratio_sweep,
        "grids": grids,
        "success_rates": success_rates,
    }
    return JSONResponse(_sanitize(payload))


# Serve the static dashboard at "/". Mounted last so /api/* takes priority.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
