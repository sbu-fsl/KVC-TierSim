"""CLI comparing KV-cache policies on a single-tier (disk) hardware model.

The policy strategies themselves live in the ``policies/`` package; this module
owns the *hardware math* (how X and Y are derived) and the experiment plumbing.
Every registered policy is evaluated automatically, so adding a policy module
makes it show up here too.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable

from src.policies import PolicyContext, evaluate, get_policies, policy_keys
from src.policies.core import PolicyResult  # re-exported for backward compatibility
from src.pool import DISKS, DRAMS, GPUS, LINKS

BYTES_PER_BLOCK = 2e6
TOKENS_PER_BLOCK = 16
MODEL_PARAMS = 8e9
GPU_ETA = 0.5

__all__ = [
    "HardwareConfig",
    "ExperimentConfig",
    "PolicyResult",
    "STACK_PRESETS",
    "hardware_throughputs",
    "run_experiments",
    "main",
]


@dataclass(frozen=True)
class HardwareConfig:
    stack_key: str = "custom"
    gpu_key: str = "A100"
    dram_key: str = "DDR5"
    disk_key: str = "NVMe"
    link_key: str = "NVLink"
    gpu_count: int = 1
    dram_count: int = 4


@dataclass(frozen=True)
class ExperimentConfig:
    total_blocks: tuple[int, ...]
    miss_blocks: tuple[int, ...]
    request_rate: float = 1.0
    p95_seconds: float = 1.0
    storage_tier: str = "disk"
    model_params: float = MODEL_PARAMS
    tokens_per_block: int = TOKENS_PER_BLOCK
    bytes_per_block: float = BYTES_PER_BLOCK
    gpu_eta: float = GPU_ETA


STACK_PRESETS = {
    "h200": HardwareConfig(
        stack_key="h200", gpu_key="H200", dram_key="DDR5", disk_key="NVMe", link_key="NVLink"
    ),
    "a5000": HardwareConfig(
        stack_key="a5000", gpu_key="A5000", dram_key="DDR5", disk_key="NVMe", link_key="NVLink"
    ),
    "a100_ddr5_nvme": HardwareConfig(
        stack_key="a100_ddr5_nvme", gpu_key="A100", dram_key="DDR5", disk_key="NVMe", link_key="NVLink"
    ),
    "a100_ddr4_nvme": HardwareConfig(
        stack_key="a100_ddr4_nvme", gpu_key="A100", dram_key="DDR4", disk_key="NVMe", link_key="NVLink"
    ),
    "a100_ddr4_sata": HardwareConfig(
        stack_key="a100_ddr4_sata", gpu_key="A100", dram_key="DDR4", disk_key="SATA", link_key="PCIe"
    ),
}


def _resolve_hardware_config(args: argparse.Namespace) -> HardwareConfig:
    defaults = HardwareConfig()
    preset = STACK_PRESETS.get(args.stack, defaults)
    return HardwareConfig(
        stack_key=args.stack or preset.stack_key,
        gpu_key=args.gpu or preset.gpu_key,
        dram_key=args.dram or preset.dram_key,
        disk_key=args.disk or preset.disk_key,
        link_key=args.link or preset.link_key,
        gpu_count=args.gpu_count if args.gpu_count is not None else preset.gpu_count,
        dram_count=args.dram_count if args.dram_count is not None else preset.dram_count,
    )


def hardware_throughputs(
    hardware: HardwareConfig, experiment: ExperimentConfig
) -> tuple[float, float]:
    """Shared hardware math: single-tier restore rate X and recompute rate Y.

    Restore from disk is two serial data-movement costs (disk read + link
    transfer). Recompute is the GPU decode throughput for the model.
    """
    gpu = GPUS[hardware.gpu_key]
    link = LINKS[hardware.link_key]
    disk = DISKS[hardware.disk_key]

    storage_throughput = 1.0 / (
        (experiment.bytes_per_block / disk.bandwidth)
        + (experiment.bytes_per_block / link.bandwidth)
    )
    t_per_token = gpu.gpu_compute_band(
        experiment.model_params, gpu_count=hardware.gpu_count, eta=experiment.gpu_eta
    )
    recompute_throughput = 1.0 / (experiment.tokens_per_block * t_per_token)
    return storage_throughput, recompute_throughput


def _speedup(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else math.inf


def run_experiments(hardware: HardwareConfig, experiment: ExperimentConfig):
    storage_throughput, recompute_throughput = hardware_throughputs(hardware, experiment)

    rows: list[dict] = []
    for total_blocks in experiment.total_blocks:
        for miss_blocks in experiment.miss_blocks:
            if miss_blocks < 0:
                raise ValueError("miss_blocks must be greater than or equal to zero")
            if miss_blocks > total_blocks:
                raise ValueError("miss_blocks cannot exceed total_blocks")

            ctx = PolicyContext(
                total_blocks=total_blocks,
                miss_blocks=miss_blocks,
                storage_throughput=storage_throughput,
                recompute_throughput=recompute_throughput,
                request_rate=experiment.request_rate,
                p95_seconds=experiment.p95_seconds,
                stack_key=hardware.stack_key,
            )
            results = {policy.name: evaluate(policy, ctx) for policy in get_policies()}

            row = {
                "stack_key": hardware.stack_key,
                "request_rate": experiment.request_rate,
                "p95_seconds": experiment.p95_seconds,
                "slo_seconds": experiment.p95_seconds,
                "total_blocks": total_blocks,
                "miss_blocks": miss_blocks,
                "hit_blocks": total_blocks - miss_blocks,
                "storage_throughput": storage_throughput,
                "recompute_throughput": recompute_throughput,
            }
            row.update({name: asdict(result) for name, result in results.items()})

            # Pairwise speedups among the canonical policies (kept for compatibility).
            aware = results.get("performance_aware")
            default = results.get("default_policy")
            all_compute = results.get("all_compute")
            if default and all_compute:
                row["speedup_default_vs_all_compute"] = _speedup(
                    default.total_time, all_compute.total_time
                )
            if all_compute and aware:
                row["speedup_all_compute_vs_performance_aware"] = _speedup(
                    all_compute.total_time, aware.total_time
                )
            if aware and default:
                row["speedup_performance_aware_vs_default"] = _speedup(
                    aware.total_time, default.total_time
                )

            rows.append(row)
    return rows


def _slo_summary(rows: list[dict], policy_key: str) -> dict:
    if not rows or not all(policy_key in row for row in rows):
        return {"status": "fail"}
    return {
        "status": "pass"
        if all(row[policy_key]["meets_slo"] for row in rows)
        else "fail"
    }


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return tuple(int(value) for value in values)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare KV cache policies on a single-tier hardware model."
    )
    parser.add_argument(
        "--stack", choices=tuple(STACK_PRESETS.keys()), default=None, help="Predefined hardware stack."
    )
    parser.add_argument("--gpu", choices=tuple(GPUS.keys()), default=None)
    parser.add_argument("--dram", choices=tuple(DRAMS.keys()), default=None)
    parser.add_argument("--disk", choices=tuple(DISKS.keys()), default=None)
    parser.add_argument("--link", choices=tuple(LINKS.keys()), default=None)
    parser.add_argument("--gpu-count", type=int, default=None)
    parser.add_argument("--dram-count", type=int, default=None)
    parser.add_argument("--storage-tier", choices=("disk", "dram"), default="disk")
    parser.add_argument("--total-blocks", default="50000", help="Comma-separated restore request sizes.")
    parser.add_argument("--miss-blocks", default="0", help="Comma-separated cache-miss block counts.")
    parser.add_argument("--request-rate", type=float, default=1.0, help="Target request rate (requests/s).")
    parser.add_argument("--p95-seconds", type=float, default=1.0, help="Target P95 latency in seconds.")
    parser.add_argument("--model-params", type=float, default=MODEL_PARAMS)
    parser.add_argument("--tokens-per-block", type=int, default=TOKENS_PER_BLOCK)
    parser.add_argument("--bytes-per-block", type=float, default=BYTES_PER_BLOCK)
    parser.add_argument("--gpu-eta", type=float, default=GPU_ETA)
    parser.add_argument("--output", default="", help="Optional JSON output file.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    hardware = _resolve_hardware_config(args)
    experiment = ExperimentConfig(
        total_blocks=_parse_int_list(args.total_blocks),
        miss_blocks=_parse_int_list(args.miss_blocks),
        request_rate=args.request_rate,
        p95_seconds=args.p95_seconds,
        storage_tier=args.storage_tier,
        model_params=args.model_params,
        tokens_per_block=args.tokens_per_block,
        bytes_per_block=args.bytes_per_block,
        gpu_eta=args.gpu_eta,
    )

    rows = run_experiments(hardware, experiment)
    payload = {
        "hardware": asdict(hardware),
        "experiment": asdict(experiment),
        "results": rows,
        "slo_summary": {key: _slo_summary(rows, key) for key in policy_keys()},
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved -> {args.output}")
    else:
        print(json.dumps(payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
