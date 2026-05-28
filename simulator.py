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
    gpu_key: str = "A100"
    dram_key: str = "DDR5"
    disk_key: str = "NVMe"
    link_key: str = "NVLink"
    gpu_count: int = 1
    dram_count: int = 4


@dataclass(frozen=True)
class ExperimentConfig:
    user_counts: tuple[int, ...]
    context_lengths: tuple[int, ...]
    request_count: int = 1
    context_unit: str = "tokens"
    storage_tier: str = "disk"
    default_policy: str = "restore_all"
    model_params: float = MODEL_PARAMS
    tokens_per_block: int = TOKENS_PER_BLOCK
    bytes_per_block: float = BYTES_PER_BLOCK
    slo_seconds: float = 1.0
    gpu_eta: float = GPU_ETA


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    users: int
    context_length: int
    context_unit: str
    request_count: int
    slo_seconds: float
    meets_slo: bool
    met_requests: int
    total_requests: int
    success_rate: float
    slo_margin: float
    total_blocks: int
    storage_blocks: float
    recompute_blocks: float
    storage_throughput: float
    recompute_throughput: float
    storage_time: float
    recompute_time: float
    total_time: float


def _hardware_stack(config: HardwareConfig):
    return {
        "gpu": GPUS[config.gpu_key],
        "dram": DRAMS[config.dram_key],
        "disk": DISKS[config.disk_key],
        "link": LINKS[config.link_key],
        "gpu_count": config.gpu_count,
        "dram_count": config.dram_count,
    }


def _blocks_for_context(context_length: int, context_unit: str, tokens_per_block: int) -> int:
    if context_unit == "blocks":
        return int(context_length)
    if context_unit != "tokens":
        raise ValueError(f"Unsupported context_unit: {context_unit}")
    return math.ceil(context_length / tokens_per_block)


def _total_blocks(users: int, context_length: int, context_unit: str, tokens_per_block: int) -> int:
    return users * _blocks_for_context(context_length, context_unit, tokens_per_block)


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


def _evaluate_split(
    total_blocks: int,
    storage_throughput: float,
    recompute_throughput: float,
    slo_seconds: float | None,
) -> tuple[float, float]:
    if total_blocks <= 0:
        return 0.0, 0.0

    balanced_storage = (storage_throughput / (storage_throughput + recompute_throughput)) * total_blocks
    balanced_recompute = total_blocks - balanced_storage

    if slo_seconds is not None:
        recompute_limit = recompute_throughput * slo_seconds
        balanced_recompute = min(balanced_recompute, recompute_limit)

    storage_blocks = total_blocks - balanced_recompute
    return storage_blocks, balanced_recompute


def _time_from_blocks(blocks: float, throughput: float) -> float:
    if throughput <= 0:
        return math.inf
    return blocks / throughput


def _requests_met_within_slo(request_count: int, service_time: float, slo_seconds: float | None) -> tuple[int, float, bool]:
    if request_count <= 0:
        return 0, 0.0, True
    if slo_seconds is None:
        return request_count, 1.0, True
    if service_time <= 0:
        return request_count, 1.0, True

    met_requests = min(request_count, int(slo_seconds / service_time))
    success_rate = met_requests / request_count
    return met_requests, success_rate, met_requests == request_count


def evaluate_performance_aware(
    total_blocks: int,
    storage_throughput: float,
    recompute_throughput: float,
    slo_seconds: float | None,
    request_count: int,
) -> PolicyResult:
    storage_blocks, recompute_blocks = _evaluate_split(
        total_blocks, storage_throughput, recompute_throughput, slo_seconds
    )
    storage_time = _time_from_blocks(storage_blocks, storage_throughput)
    recompute_time = _time_from_blocks(recompute_blocks, recompute_throughput)
    total_time = max(storage_time, recompute_time)
    met_requests, success_rate, meets_slo = _requests_met_within_slo(request_count, total_time, slo_seconds)
    return PolicyResult(
        policy="performance_aware",
        users=0,
        context_length=0,
        context_unit="tokens",
        request_count=request_count,
        slo_seconds=float(slo_seconds or 0.0),
        meets_slo=meets_slo,
        met_requests=met_requests,
        total_requests=request_count,
        success_rate=success_rate,
        slo_margin=(float(slo_seconds) - total_time) if slo_seconds is not None else math.nan,
        total_blocks=total_blocks,
        storage_blocks=storage_blocks,
        recompute_blocks=recompute_blocks,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        storage_time=storage_time,
        recompute_time=recompute_time,
        total_time=total_time,
    )


def evaluate_default_policy(
    total_blocks: int,
    storage_throughput: float,
    recompute_throughput: float,
    policy: str,
    slo_seconds: float | None,
    request_count: int,
) -> PolicyResult:
    if policy == "restore_all":
        storage_blocks = float(total_blocks)
        recompute_blocks = 0.0
    elif policy == "recompute_all":
        storage_blocks = 0.0
        recompute_blocks = float(total_blocks)
    elif policy == "balanced":
        storage_blocks, recompute_blocks = _evaluate_split(
            total_blocks, storage_throughput, recompute_throughput, slo_seconds=None
        )
    else:
        raise ValueError(f"Unsupported default_policy: {policy}")

    storage_time = _time_from_blocks(storage_blocks, storage_throughput)
    recompute_time = _time_from_blocks(recompute_blocks, recompute_throughput)
    total_time = max(storage_time, recompute_time)
    met_requests, success_rate, meets_slo = _requests_met_within_slo(request_count, total_time, slo_seconds)
    return PolicyResult(
        policy=policy,
        users=0,
        context_length=0,
        context_unit="tokens",
        request_count=request_count,
        slo_seconds=float(slo_seconds or 0.0),
        meets_slo=meets_slo,
        met_requests=met_requests,
        total_requests=request_count,
        success_rate=success_rate,
        slo_margin=(float(slo_seconds) - total_time) if slo_seconds is not None else math.nan,
        total_blocks=total_blocks,
        storage_blocks=storage_blocks,
        recompute_blocks=recompute_blocks,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        storage_time=storage_time,
        recompute_time=recompute_time,
        total_time=total_time,
    )


def _slo_summary(rows: list[dict], policy_key: str) -> dict:
    if not rows:
        return {"total_requests": 0, "met_requests": 0, "rate": 0.0}

    met = sum(row[policy_key]["met_requests"] for row in rows)
    total = sum(row[policy_key]["total_requests"] for row in rows)
    return {
        "total_requests": total,
        "met": met,
        "met_requests": met,
        "rate": (met / total) if total > 0 else 0.0,
    }


def run_experiments(hardware: HardwareConfig, experiment: ExperimentConfig):
    storage_throughput = _storage_throughput(hardware, experiment.storage_tier, experiment.bytes_per_block)
    recompute_throughput = _recompute_throughput(
        hardware, experiment.model_params, experiment.tokens_per_block, experiment.gpu_eta
    )

    rows: list[dict] = []
    for users in experiment.user_counts:
        for context_length in experiment.context_lengths:
            total_blocks = _total_blocks(
                users, context_length, experiment.context_unit, experiment.tokens_per_block
            )

            aware = evaluate_performance_aware(
                total_blocks,
                storage_throughput,
                recompute_throughput,
                experiment.slo_seconds,
                experiment.request_count,
            )
            default = evaluate_default_policy(
                total_blocks,
                storage_throughput,
                recompute_throughput,
                experiment.default_policy,
                experiment.slo_seconds,
                experiment.request_count,
            )

            aware = PolicyResult(
                **{**asdict(aware), "users": users, "context_length": context_length, "context_unit": experiment.context_unit}
            )
            default = PolicyResult(
                **{**asdict(default), "users": users, "context_length": context_length, "context_unit": experiment.context_unit}
            )

            rows.append(
                {
                    "users": users,
                    "context_length": context_length,
                    "context_unit": experiment.context_unit,
                    "request_count": experiment.request_count,
                    "total_blocks": total_blocks,
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
    parser.add_argument("--users", default="1,2,4,8", help="Comma-separated user counts.")
    parser.add_argument("--contexts", default="16000,64000,128000,256000", help="Comma-separated context lengths.")
    parser.add_argument("--requests", type=int, default=1, help="Number of requests per user/context scenario.")
    parser.add_argument("--context-unit", choices=("tokens", "blocks"), default="tokens")
    parser.add_argument("--storage-tier", choices=("disk", "dram"), default="disk")
    parser.add_argument("--default-policy", choices=("restore_all", "recompute_all", "balanced"), default="restore_all")
    parser.add_argument("--slo-seconds", type=float, default=1.0)
    parser.add_argument("--gpu", choices=tuple(GPUS.keys()), default="A100")
    parser.add_argument("--dram", choices=tuple(DRAMS.keys()), default="DDR5")
    parser.add_argument("--disk", choices=tuple(DISKS.keys()), default="NVMe")
    parser.add_argument("--link", choices=tuple(LINKS.keys()), default="NVLink")
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--dram-count", type=int, default=4)
    parser.add_argument("--model-params", type=float, default=MODEL_PARAMS)
    parser.add_argument("--tokens-per-block", type=int, default=TOKENS_PER_BLOCK)
    parser.add_argument("--bytes-per-block", type=float, default=BYTES_PER_BLOCK)
    parser.add_argument("--gpu-eta", type=float, default=GPU_ETA)
    parser.add_argument("--output", default="", help="Optional JSON output file.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    hardware = HardwareConfig(
        gpu_key=args.gpu,
        dram_key=args.dram,
        disk_key=args.disk,
        link_key=args.link,
        gpu_count=args.gpu_count,
        dram_count=args.dram_count,
    )
    experiment = ExperimentConfig(
        user_counts=_parse_int_list(args.users),
        context_lengths=_parse_int_list(args.contexts),
        request_count=args.requests,
        context_unit=args.context_unit,
        storage_tier=args.storage_tier,
        default_policy=args.default_policy,
        model_params=args.model_params,
        tokens_per_block=args.tokens_per_block,
        bytes_per_block=args.bytes_per_block,
        slo_seconds=args.slo_seconds,
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
