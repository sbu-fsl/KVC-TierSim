from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable

from src.pool import DISKS, DRAMS, GPUS, LINKS


BYTES_PER_BLOCK = 2e6
TOKENS_PER_BLOCK = 16
MODEL_PARAMS = 8e9
GPU_ETA = 0.5


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


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    stack_key: str
    total_blocks: int
    miss_blocks: int
    hit_blocks: int
    request_rate: float
    p95_seconds: float
    slo_seconds: float
    meets_slo: bool
    success_rate: float
    slo_margin: float
    storage_blocks: float
    recompute_blocks: float
    storage_throughput: float
    recompute_throughput: float
    storage_time: float
    recompute_time: float
    total_time: float


STACK_PRESETS = {
    "h200": HardwareConfig(stack_key="h200", gpu_key="H200", dram_key="DDR5", disk_key="NVMe", link_key="NVLink"),
    "a5000": HardwareConfig(stack_key="a5000", gpu_key="A5000", dram_key="DDR5", disk_key="NVMe", link_key="NVLink"),
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


def _recompute_throughput(hardware: HardwareConfig, model_params: float, tokens_per_block: int, gpu_eta: float) -> float:
    gpu = GPUS[hardware.gpu_key]
    t_per_token = gpu.gpu_compute_band(model_params, gpu_count=hardware.gpu_count, eta=gpu_eta)
    return 1.0 / (tokens_per_block * t_per_token)


def _storage_throughput(hardware: HardwareConfig, storage_tier: str, bytes_per_block: float) -> float:
    link = LINKS[hardware.link_key]
    if storage_tier == "disk":
        tier_bandwidth = DISKS[hardware.disk_key].bandwidth
    elif storage_tier == "dram":
        tier_bandwidth = DRAMS[hardware.dram_key].bandwidth * hardware.dram_count
    else:
        raise ValueError(f"Unsupported storage_tier: {storage_tier}")
    return min(tier_bandwidth, link.bandwidth) / bytes_per_block


def _effective_deadline(request_rate: float, p95_seconds: float) -> float:
    if request_rate <= 0:
        raise ValueError("request_rate must be greater than zero")
    if p95_seconds <= 0:
        raise ValueError("p95_seconds must be greater than zero")
    return min(p95_seconds, 1.0 / request_rate)


def _time_from_blocks(blocks: float, throughput: float) -> float:
    if throughput <= 0:
        return math.inf
    return blocks / throughput


def _slo_outcome(service_time: float, slo_seconds: float | None) -> tuple[float, bool]:
    if slo_seconds is None:
        return 1.0, True
    if service_time <= slo_seconds:
        return 1.0, True
    return 0.0, False


def _balanced_recompute_blocks(total_blocks: int, miss_blocks: int, storage_throughput: float, recompute_throughput: float) -> float:
    if total_blocks <= 0:
        return 0.0
    if recompute_throughput <= 0:
        return float(miss_blocks)
    if storage_throughput <= 0:
        return float(total_blocks)

    balanced_recompute = (total_blocks * recompute_throughput) / (storage_throughput + recompute_throughput)
    return max(float(miss_blocks), min(float(total_blocks), balanced_recompute))


def _build_policy_result(
    policy: str,
    stack_key: str,
    total_blocks: int,
    miss_blocks: int,
    storage_blocks: float,
    recompute_blocks: float,
    storage_throughput: float,
    recompute_throughput: float,
    request_rate: float,
    p95_seconds: float,
) -> PolicyResult:
    storage_time = _time_from_blocks(storage_blocks, storage_throughput)
    recompute_time = _time_from_blocks(recompute_blocks, recompute_throughput)
    total_time = max(storage_time, recompute_time)
    slo_seconds = _effective_deadline(request_rate, p95_seconds)
    success_rate, meets_slo = _slo_outcome(total_time, slo_seconds)

    return PolicyResult(
        policy=policy,
        stack_key=stack_key,
        total_blocks=total_blocks,
        miss_blocks=miss_blocks,
        hit_blocks=max(0, total_blocks - miss_blocks),
        request_rate=request_rate,
        p95_seconds=p95_seconds,
        slo_seconds=slo_seconds,
        meets_slo=meets_slo,
        success_rate=success_rate,
        slo_margin=slo_seconds - total_time,
        storage_blocks=storage_blocks,
        recompute_blocks=recompute_blocks,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        storage_time=storage_time,
        recompute_time=recompute_time,
        total_time=total_time,
    )


def evaluate_performance_aware(
    total_blocks: int,
    miss_blocks: int,
    storage_throughput: float,
    recompute_throughput: float,
    request_rate: float,
    p95_seconds: float,
    stack_key: str,
) -> PolicyResult:
    recompute_blocks = _balanced_recompute_blocks(total_blocks, miss_blocks, storage_throughput, recompute_throughput)
    storage_blocks = float(total_blocks) - recompute_blocks
    return _build_policy_result(
        policy="performance_aware",
        stack_key=stack_key,
        total_blocks=total_blocks,
        miss_blocks=miss_blocks,
        storage_blocks=storage_blocks,
        recompute_blocks=recompute_blocks,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        request_rate=request_rate,
        p95_seconds=p95_seconds,
    )


def evaluate_default_policy(
    total_blocks: int,
    miss_blocks: int,
    storage_throughput: float,
    recompute_throughput: float,
    request_rate: float,
    p95_seconds: float,
    stack_key: str,
) -> PolicyResult:
    hit_blocks = max(0, total_blocks - miss_blocks)
    return _build_policy_result(
        policy="restore_hits_recompute_misses",
        stack_key=stack_key,
        total_blocks=total_blocks,
        miss_blocks=miss_blocks,
        storage_blocks=float(hit_blocks),
        recompute_blocks=float(miss_blocks),
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        request_rate=request_rate,
        p95_seconds=p95_seconds,
    )


def _slo_summary(rows: list[dict], policy_key: str) -> dict:
    if not rows:
        return {"status": "fail"}

    return {"status": "pass" if all(row[policy_key]["meets_slo"] for row in rows) else "fail"}


def run_experiments(hardware: HardwareConfig, experiment: ExperimentConfig):
    storage_throughput = _storage_throughput(hardware, experiment.storage_tier, experiment.bytes_per_block)
    recompute_throughput = _recompute_throughput(
        hardware, experiment.model_params, experiment.tokens_per_block, experiment.gpu_eta
    )

    rows: list[dict] = []
    for total_blocks in experiment.total_blocks:
        for miss_blocks in experiment.miss_blocks:
            if miss_blocks < 0:
                raise ValueError("miss_blocks must be greater than or equal to zero")
            if miss_blocks > total_blocks:
                raise ValueError("miss_blocks cannot exceed total_blocks")

            aware = evaluate_performance_aware(
                total_blocks,
                miss_blocks,
                storage_throughput,
                recompute_throughput,
                experiment.request_rate,
                experiment.p95_seconds,
                hardware.stack_key,
            )
            default = evaluate_default_policy(
                total_blocks,
                miss_blocks,
                storage_throughput,
                recompute_throughput,
                experiment.request_rate,
                experiment.p95_seconds,
                hardware.stack_key,
            )

            rows.append(
                {
                    "stack_key": hardware.stack_key,
                    "request_rate": experiment.request_rate,
                    "p95_seconds": experiment.p95_seconds,
                    "slo_seconds": aware.slo_seconds,
                    "total_blocks": total_blocks,
                    "miss_blocks": miss_blocks,
                    "hit_blocks": total_blocks - miss_blocks,
                    "storage_throughput": storage_throughput,
                    "recompute_throughput": recompute_throughput,
                    "performance_aware": asdict(aware),
                    "default_policy": asdict(default),
                    "speedup": default.total_time / aware.total_time if aware.total_time > 0 else math.inf,
                }
            )
    return rows


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return tuple(int(value) for value in values)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare performance-aware and default KV cache policies.")
    parser.add_argument("--stack", choices=tuple(STACK_PRESETS.keys()), default=None, help="Predefined hardware stack.")
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
        "slo_summary": {
            "performance_aware": _slo_summary(rows, "performance_aware"),
            "default_policy": _slo_summary(rows, "default_policy"),
        },
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