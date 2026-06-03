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
    reassigned_hit_blocks: float
    request_rate: float
    p95_seconds: float
    slo_seconds: float
    meets_slo: bool
    success_rate: float
    slo_margin: float
    decision_mode: str
    latency_ratio: float
    gpu_utilization: float
    storage_utilization: float
    latency_violation: float
    gpu_violation: float
    storage_violation: float
    max_violation: float
    storage_blocks: float
    recompute_blocks: float
    storage_throughput: float
    recompute_throughput: float
    storage_time: float
    recompute_time: float
    total_time: float


STACK_PRESETS = {
    "h200": HardwareConfig(
        stack_key="h200",
        gpu_key="H200",
        dram_key="DDR5",
        disk_key="NVMe",
        link_key="NVLink",
    ),
    "a5000": HardwareConfig(
        stack_key="a5000",
        gpu_key="A5000",
        dram_key="DDR5",
        disk_key="NVMe",
        link_key="NVLink",
    ),
    "a100_ddr5_nvme": HardwareConfig(
        stack_key="a100_ddr5_nvme",
        gpu_key="A100",
        dram_key="DDR5",
        disk_key="NVMe",
        link_key="NVLink",
    ),
    "a100_ddr4_nvme": HardwareConfig(
        stack_key="a100_ddr4_nvme",
        gpu_key="A100",
        dram_key="DDR4",
        disk_key="NVMe",
        link_key="NVLink",
    ),
    "a100_ddr4_sata": HardwareConfig(
        stack_key="a100_ddr4_sata",
        gpu_key="A100",
        dram_key="DDR4",
        disk_key="SATA",
        link_key="PCIe",
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
        dram_count=args.dram_count
        if args.dram_count is not None
        else preset.dram_count,
    )


def _simulator_compute_time(
    hardware: HardwareConfig,
    blocks: float,
    model_params: float,
    tokens_per_block: int,
    gpu_eta: float,
) -> float:
    """Match main.py::build_compute_curve for one block count."""
    if blocks <= 0:
        return 0.0

    gpu = GPUS[hardware.gpu_key]
    link = LINKS[hardware.link_key]

    model_size_bytes = model_params * 2  # bf16: 2 bytes per parameter
    vram_cap = gpu.hbm_capacity * hardware.gpu_count - model_size_bytes
    if vram_cap <= 0:
        return math.inf

    vram_blocks = vram_cap / BYTES_PER_BLOCK
    t_per_token = gpu.gpu_compute_band(
        model_params, gpu_count=hardware.gpu_count, eta=gpu_eta
    )

    def block_time(n: float, bandwidth: float) -> float:
        t_mem = (n * BYTES_PER_BLOCK) / bandwidth
        t_cmp = n * tokens_per_block * t_per_token
        return max(t_mem, t_cmp)

    if blocks <= vram_blocks:
        return block_time(blocks, gpu.hbm_bandwidth)

    t_vram = block_time(vram_blocks, gpu.hbm_bandwidth)
    t_overflow = block_time(blocks - vram_blocks, link.bandwidth)
    return t_vram + t_overflow


def _simulator_restore_time(
    hardware: HardwareConfig,
    blocks: float,
    model_params: float,
    bytes_per_block: float,
    tokens_per_block: int,
    gpu_eta: float,
) -> float:
    """Return storage-tier restore time for disk-resident hit blocks.

    The policy is applied only to cache-hit blocks that were evicted to the
    storage tier. Hits in GPU HBM or CPU DRAM are assumed to be served without a
    restoration/recomputation decision. Therefore, the restore path here models
    only the disk-resident data movement cost: disk read overhead plus link
    transfer overhead. No GPU recomputation term is included in restoration.
    """
    if blocks <= 0:
        return 0.0

    disk = DISKS[hardware.disk_key]
    link = LINKS[hardware.link_key]
    bytes_to_restore = blocks * bytes_per_block
    return bytes_to_restore / disk.bandwidth + bytes_to_restore / link.bandwidth


def _balanced_reassigned_hit_blocks(
    total_blocks: int,
    miss_blocks: int,
    storage_throughput: float,
    recompute_throughput: float,
) -> float:
    """Compute k from the policy equation.

    N is total requested blocks, M is missing blocks, and k is the number of
    cache-hit blocks reassigned from restoration to recomputation:

        k = max(0, min(N-M, (N*X)/(X+Y) - M))

    Here X and Y are represented in blocks/s. This is equivalent to bytes/s
    because every block has the same size B.
    """
    if total_blocks <= 0:
        return 0.0
    if miss_blocks < 0 or miss_blocks > total_blocks:
        raise ValueError("miss_blocks must be in the range [0, total_blocks]")

    hit_blocks = total_blocks - miss_blocks
    if hit_blocks <= 0:
        return 0.0
    if recompute_throughput <= 0:
        return 0.0
    if storage_throughput <= 0:
        return float(hit_blocks)

    k = (total_blocks * recompute_throughput) / (
        recompute_throughput + storage_throughput
    ) - miss_blocks
    return max(0.0, min(float(hit_blocks), k))


def _build_policy_result(
    policy: str,
    stack_key: str,
    total_blocks: int,
    miss_blocks: int,
    storage_time: float,
    recompute_time: float,
    storage_blocks: float,
    recompute_blocks: float,
    reassigned_hit_blocks: float,
    storage_throughput: float,
    recompute_throughput: float,
    request_rate: float,
    p95_seconds: float,
    decision_mode: str,
) -> PolicyResult:
    total_time = max(storage_time, recompute_time)
    if request_rate <= 0:
        raise ValueError("request_rate must be greater than zero")
    if p95_seconds <= 0:
        raise ValueError("p95_seconds must be greater than zero")

    # Policy constraints:
    # T_req <= p, r*T_c <= 1, and r*T_s <= 1.
    latency_ratio = total_time / p95_seconds
    gpu_utilization = request_rate * recompute_time
    storage_utilization = request_rate * storage_time

    latency_violation = max(0.0, latency_ratio - 1.0)
    gpu_violation = max(0.0, gpu_utilization - 1.0)
    storage_violation = max(0.0, storage_utilization - 1.0)
    max_violation = max(latency_violation, gpu_violation, storage_violation)

    meets_slo = max_violation == 0.0
    success_rate = 1.0 if meets_slo else 0.0

    return PolicyResult(
        policy=policy,
        stack_key=stack_key,
        total_blocks=total_blocks,
        miss_blocks=miss_blocks,
        hit_blocks=max(0, total_blocks - miss_blocks),
        reassigned_hit_blocks=reassigned_hit_blocks,
        request_rate=request_rate,
        p95_seconds=p95_seconds,
        slo_seconds=p95_seconds,
        meets_slo=meets_slo,
        success_rate=success_rate,
        slo_margin=p95_seconds - total_time,
        decision_mode=decision_mode,
        latency_ratio=latency_ratio,
        gpu_utilization=gpu_utilization,
        storage_utilization=storage_utilization,
        latency_violation=latency_violation,
        gpu_violation=gpu_violation,
        storage_violation=storage_violation,
        max_violation=max_violation,
        storage_blocks=storage_blocks,
        recompute_blocks=recompute_blocks,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        storage_time=storage_time,
        recompute_time=recompute_time,
        total_time=total_time,
    )


def _constraint_max_violation(
    storage_time: float,
    recompute_time: float,
    request_rate: float,
    p95_seconds: float,
) -> float:
    total_time = max(storage_time, recompute_time)
    if p95_seconds <= 0:
        return math.inf

    latency_violation = max(0.0, (total_time / p95_seconds) - 1.0)
    gpu_violation = max(0.0, request_rate * recompute_time - 1.0)
    storage_violation = max(0.0, request_rate * storage_time - 1.0)
    return max(latency_violation, gpu_violation, storage_violation)


def _allocation_times(
    total_blocks: int,
    miss_blocks: int,
    reassigned_hit_blocks: float,
    storage_throughput: float,
    recompute_throughput: float,
) -> tuple[float, float, float, float]:
    hit_blocks = max(0, total_blocks - miss_blocks)
    k = max(0.0, min(float(hit_blocks), reassigned_hit_blocks))

    recompute_blocks = float(miss_blocks) + k
    storage_blocks = float(hit_blocks) - k

    storage_time = (
        storage_blocks / storage_throughput if storage_throughput > 0 else math.inf
    )
    recompute_time = (
        recompute_blocks / recompute_throughput
        if recompute_throughput > 0
        else math.inf
    )
    return storage_blocks, recompute_blocks, storage_time, recompute_time


def _best_effort_reassigned_hit_blocks(
    total_blocks: int,
    miss_blocks: int,
    storage_throughput: float,
    recompute_throughput: float,
    request_rate: float,
    p95_seconds: float,
) -> float:
    hit_blocks = max(0, total_blocks - miss_blocks)
    if hit_blocks <= 0:
        return 0.0

    candidates = {0.0, float(hit_blocks)}
    balanced_k = _balanced_reassigned_hit_blocks(
        total_blocks, miss_blocks, storage_throughput, recompute_throughput
    )
    candidates.add(balanced_k)

    # Candidate k values from where individual constraints can become tight.
    if recompute_throughput > 0:
        candidates.add((recompute_throughput / request_rate) - miss_blocks)
        candidates.add(p95_seconds * recompute_throughput - miss_blocks)
    if storage_throughput > 0:
        candidates.add(total_blocks - miss_blocks - (storage_throughput / request_rate))
        candidates.add(total_blocks - miss_blocks - (p95_seconds * storage_throughput))

    valid_candidates: list[float] = []
    for k in candidates:
        valid_candidates.append(max(0.0, min(float(hit_blocks), float(k))))

    # Evaluate all clamped candidate points and use a deterministic tie-break.
    best_k = 0.0
    best_score = math.inf
    best_total_time = math.inf
    for k in valid_candidates:
        _, _, storage_time, recompute_time = _allocation_times(
            total_blocks,
            miss_blocks,
            reassigned_hit_blocks=k,
            storage_throughput=storage_throughput,
            recompute_throughput=recompute_throughput,
        )
        score = _constraint_max_violation(
            storage_time=storage_time,
            recompute_time=recompute_time,
            request_rate=request_rate,
            p95_seconds=p95_seconds,
        )
        total_time = max(storage_time, recompute_time)
        if score < best_score or (score == best_score and total_time < best_total_time):
            best_score = score
            best_total_time = total_time
            best_k = k

    return best_k


def evaluate_performance_aware(
    hardware: HardwareConfig,
    total_blocks: int,
    miss_blocks: int,
    storage_throughput: float,
    recompute_throughput: float,
    request_rate: float,
    p95_seconds: float,
) -> PolicyResult:
    reassigned_hit_blocks = _balanced_reassigned_hit_blocks(
        total_blocks, miss_blocks, storage_throughput, recompute_throughput
    )
    storage_blocks, recompute_blocks, storage_time, recompute_time = _allocation_times(
        total_blocks,
        miss_blocks,
        reassigned_hit_blocks=reassigned_hit_blocks,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
    )

    result = _build_policy_result(
        policy="performance_aware",
        stack_key=hardware.stack_key,
        total_blocks=total_blocks,
        miss_blocks=miss_blocks,
        storage_time=storage_time,
        recompute_time=recompute_time,
        storage_blocks=storage_blocks,
        recompute_blocks=recompute_blocks,
        reassigned_hit_blocks=reassigned_hit_blocks,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        request_rate=request_rate,
        p95_seconds=p95_seconds,
        decision_mode="balanced",
    )

    if result.meets_slo:
        return result

    reassigned_hit_blocks = _best_effort_reassigned_hit_blocks(
        total_blocks=total_blocks,
        miss_blocks=miss_blocks,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        request_rate=request_rate,
        p95_seconds=p95_seconds,
    )
    storage_blocks, recompute_blocks, storage_time, recompute_time = _allocation_times(
        total_blocks,
        miss_blocks,
        reassigned_hit_blocks=reassigned_hit_blocks,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
    )
    return _build_policy_result(
        policy="performance_aware",
        stack_key=hardware.stack_key,
        total_blocks=total_blocks,
        miss_blocks=miss_blocks,
        storage_time=storage_time,
        recompute_time=recompute_time,
        storage_blocks=storage_blocks,
        recompute_blocks=recompute_blocks,
        reassigned_hit_blocks=reassigned_hit_blocks,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        request_rate=request_rate,
        p95_seconds=p95_seconds,
        decision_mode="best_effort",
    )


def evaluate_default_policy(
    hardware: HardwareConfig,
    total_blocks: int,
    miss_blocks: int,
    storage_throughput: float,
    recompute_throughput: float,
    request_rate: float,
    p95_seconds: float,
) -> PolicyResult:
    storage_blocks, recompute_blocks, storage_time, recompute_time = _allocation_times(
        total_blocks,
        miss_blocks,
        reassigned_hit_blocks=0.0,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
    )
    return _build_policy_result(
        policy="restore_hits_recompute_misses",
        stack_key=hardware.stack_key,
        total_blocks=total_blocks,
        miss_blocks=miss_blocks,
        storage_time=storage_time,
        recompute_time=recompute_time,
        storage_blocks=storage_blocks,
        recompute_blocks=recompute_blocks,
        reassigned_hit_blocks=0.0,
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        request_rate=request_rate,
        p95_seconds=p95_seconds,
        decision_mode="fixed_default",
    )


def evaluate_all_compute(
    hardware: HardwareConfig,
    total_blocks: int,
    miss_blocks: int,
    storage_throughput: float,
    recompute_throughput: float,
    request_rate: float,
    p95_seconds: float,
) -> PolicyResult:
    storage_blocks, recompute_blocks, storage_time, recompute_time = _allocation_times(
        total_blocks,
        miss_blocks,
        reassigned_hit_blocks=float(max(0, total_blocks - miss_blocks)),
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
    )
    return _build_policy_result(
        policy="all_compute",
        stack_key=hardware.stack_key,
        total_blocks=total_blocks,
        miss_blocks=miss_blocks,
        storage_time=storage_time,
        recompute_time=recompute_time,
        storage_blocks=storage_blocks,
        recompute_blocks=recompute_blocks,
        reassigned_hit_blocks=float(max(0, total_blocks - miss_blocks)),
        storage_throughput=storage_throughput,
        recompute_throughput=recompute_throughput,
        request_rate=request_rate,
        p95_seconds=p95_seconds,
        decision_mode="fixed_all_compute",
    )


def _slo_summary(rows: list[dict], policy_key: str) -> dict:
    if not rows:
        return {"status": "fail"}

    return {
        "status": "pass"
        if all(row[policy_key]["meets_slo"] for row in rows)
        else "fail"
    }


def run_experiments(hardware: HardwareConfig, experiment: ExperimentConfig):
    gpu = GPUS[hardware.gpu_key]
    link = LINKS[hardware.link_key]
    # Restore from disk is modeled as two serial data-movement costs:
    # disk read overhead plus link transfer overhead.
    storage_throughput = 1.0 / (
        (experiment.bytes_per_block / DISKS[hardware.disk_key].bandwidth)
        + (experiment.bytes_per_block / link.bandwidth)
    )
    t_per_token = gpu.gpu_compute_band(
        experiment.model_params, gpu_count=hardware.gpu_count, eta=experiment.gpu_eta
    )
    recompute_throughput = 1.0 / (experiment.tokens_per_block * t_per_token)

    rows: list[dict] = []
    for total_blocks in experiment.total_blocks:
        for miss_blocks in experiment.miss_blocks:
            if miss_blocks < 0:
                raise ValueError("miss_blocks must be greater than or equal to zero")
            if miss_blocks > total_blocks:
                raise ValueError("miss_blocks cannot exceed total_blocks")

            aware = evaluate_performance_aware(
                hardware,
                total_blocks,
                miss_blocks,
                storage_throughput,
                recompute_throughput,
                experiment.request_rate,
                experiment.p95_seconds,
            )
            default = evaluate_default_policy(
                hardware,
                total_blocks,
                miss_blocks,
                storage_throughput,
                recompute_throughput,
                experiment.request_rate,
                experiment.p95_seconds,
            )
            all_compute = evaluate_all_compute(
                hardware,
                total_blocks,
                miss_blocks,
                storage_throughput,
                recompute_throughput,
                experiment.request_rate,
                experiment.p95_seconds,
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
                    "all_compute": asdict(all_compute),
                    "speedup_default_vs_all_compute": default.total_time
                    / all_compute.total_time
                    if all_compute.total_time > 0
                    else math.inf,
                    "speedup_all_compute_vs_performance_aware": all_compute.total_time
                    / aware.total_time
                    if aware.total_time > 0
                    else math.inf,
                    "speedup_performance_aware_vs_default": aware.total_time
                    / default.total_time
                    if default.total_time > 0
                    else math.inf,
                }
            )
    return rows


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return tuple(int(value) for value in values)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare performance-aware and default KV cache policies."
    )
    parser.add_argument(
        "--stack",
        choices=tuple(STACK_PRESETS.keys()),
        default=None,
        help="Predefined hardware stack.",
    )

    # X, Y
    parser.add_argument("--gpu", choices=tuple(GPUS.keys()), default=None)
    parser.add_argument("--dram", choices=tuple(DRAMS.keys()), default=None)
    parser.add_argument("--disk", choices=tuple(DISKS.keys()), default=None)
    parser.add_argument("--link", choices=tuple(LINKS.keys()), default=None)
    parser.add_argument("--gpu-count", type=int, default=None)
    parser.add_argument("--dram-count", type=int, default=None)
    parser.add_argument("--storage-tier", choices=("disk", "dram"), default="disk")

    # N, M
    parser.add_argument(
        "--total-blocks", default="50000", help="Comma-separated restore request sizes."
    )
    parser.add_argument(
        "--miss-blocks", default="0", help="Comma-separated cache-miss block counts."
    )

    # r, p
    parser.add_argument(
        "--request-rate",
        type=float,
        default=1.0,
        help="Target request rate (requests/s).",
    )
    parser.add_argument(
        "--p95-seconds", type=float, default=1.0, help="Target P95 latency in seconds."
    )

    parser.add_argument("--model-params", type=float, default=MODEL_PARAMS)
    parser.add_argument("--tokens-per-block", type=int, default=TOKENS_PER_BLOCK)

    # B
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
            "all_compute": _slo_summary(rows, "all_compute"),
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
