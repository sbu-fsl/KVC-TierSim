"""Scheduler-overhead sweep: what each policy *costs to run*.

Every other tool in this repo measures the outcome of a policy decision (latency,
SLO pass/fail). This one measures the decision itself: the wall-clock time a
policy spends inside ``decide()`` choosing how many hit blocks to recompute.

The sweep walks a request stream from micro (a single request) to macro
(millions of requests) and, for every registered policy, reports:

* total scheduler overhead    - seconds of CPU spent deciding for R requests
* per-decision latency        - the amortized cost of one decision (microseconds)
* relative overhead           - scheduler time as a percentage of the modeled
                                end-to-end inference time for the same requests

The third number is the real-world one: it says how much of a request's service
time the scheduler itself eats.

Requests are drawn from a seeded pool of randomized workload *combinations*, so
the measurement covers both of the IO-aware policy's paths (fast balanced split
and best-effort fallback) in realistic proportions, not just its cheapest branch.
Every combination is then timed on its own, which turns the run into a summary
that can be quoted directly: across N combinations, policy P added X microseconds
per decision (median and mean) over the policies that make no decision at all.
``default_policy`` and ``all_compute`` return a fixed k, so their cost is pure
dispatch - they are the baseline the overhead is measured from, and subtracting
them also cancels this module's own measurement loop out of the result.

Output is one JSON results file and four vector PDFs.

    python overhead.py --gpu A100 --output-figure-prefix a100_overhead \
      --output-results a100_overhead_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import ticker

from simulator import ExperimentConfig, HardwareConfig, hardware_throughputs
from src.policies import PolicyBase, PolicyContext, evaluate, get_policies
from src.pool import DISKS, DRAMS, GPUS, LINKS

# Default hardware. Edit here or override on the CLI.
STACK_KEY = "a100"
GPU_KEY = "A100"
DRAM_KEY = "DDR5"
DISK_KEY = "NVMe"
LINK_KEY = "NVLink"
GPU_COUNT = 1
DRAM_COUNT = 4

# Request-count sweep: micro (one request) to macro (a million), ~3 points/decade.
REQUEST_COUNTS = (
    1,
    3,
    10,
    32,
    100,
    316,
    1_000,
    3_162,
    10_000,
    31_623,
    100_000,
    316_228,
    1_000_000,
)

# Workload pool the request stream is drawn from (log-uniform where noted).
POOL_SIZE = 512
POOL_SEED = 20240517
BLOCKS_RANGE = (1_000, 200_000)  # log-uniform total blocks per request
HIT_RATIO_RANGE = (0.0, 1.0)  # uniform share of blocks that are cache hits
REQUEST_RATE_RANGE = (0.01, 2.0)  # log-uniform req/s
P95_RANGE = (1.0, 60.0)  # uniform seconds

# Timing controls. More trials at small R, where a single run is mostly noise;
# the reported number is the median trial.
WARMUP_DECISIONS = 20_000
TRIAL_SCHEDULE = ((10, 400), (1_000, 60), (100_000, 8), (None, 3))

# Per-combination benchmark: every workload in the pool is timed on its own so
# the summary can report a distribution (median / mean / p95) across combinations
# rather than a single blended number.
COMBINATION_REPS = 1_000
COMBINATION_ROUNDS = 3

# Policies that make no decision at all (fixed k). Their cost is pure dispatch,
# so they define the floor that every other policy's overhead is measured from.
BASELINE_POLICIES = ("default_policy", "all_compute")

# Plot controls (kept in step with sweep.py so figures compose in a paper).
OUTPUT_FIGURE = "a100_overhead"
OUTPUT_RESULTS = "a100_overhead_results.json"
FIG_WIDTH = 4.0
FIG_HEIGHT = 2.4
DPI = 300
FONT_SIZE = 10
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 10
LEGEND_FONT_SIZE = 8
BACKGROUND_COLOR = "#ffffff"
GRID_COLOR = "#d9d9d9"
LINE_WIDTH = 1.8
MARKER_SIZE = 3.5


@dataclass(frozen=True)
class OverheadConfig:
    """Everything the sweep needs that is not hardware."""

    request_counts: tuple[int, ...] = REQUEST_COUNTS
    pool_size: int = POOL_SIZE
    seed: int = POOL_SEED
    warmup_decisions: int = WARMUP_DECISIONS
    combination_reps: int = COMBINATION_REPS
    combination_rounds: int = COMBINATION_ROUNDS
    baseline_policies: tuple[str, ...] = BASELINE_POLICIES


@dataclass(frozen=True)
class PolicyProfile:
    """Static per-policy facts measured once over the whole workload pool."""

    name: str
    label: str
    color: str
    # Mean modeled end-to-end time of a request under this policy (seconds).
    mean_service_seconds: float
    # How often each decision_mode fired across the pool (branch mix).
    decision_modes: dict[str, int] = field(default_factory=dict)
    # True when the policy is used as a no-logic floor for the overhead deltas.
    is_baseline: bool = False


@dataclass(frozen=True)
class PoolTrace:
    """Per-combination detail behind a profile, kept out of the JSON summary."""

    # Modeled end-to-end request time, one entry per workload combination.
    service_seconds: list[float]
    # Which branch the policy took, one entry per workload combination.
    decision_modes: list[str]


def _log_uniform(rng: random.Random, low: float, high: float) -> float:
    """Draw uniformly in log space so a range spanning decades is covered evenly."""
    return low * (high / low) ** rng.random()


def _build_context_pool(
    hardware: HardwareConfig,
    config: OverheadConfig,
) -> list[PolicyContext]:
    """A fixed pool of distinct requests the timed stream cycles through.

    Contexts are built once and reused so that object construction never lands
    inside a timed region, and so a million-request run stays O(pool) in memory.
    """
    storage_throughput, recompute_throughput = hardware_throughputs(
        hardware, ExperimentConfig(total_blocks=(1,), miss_blocks=(0,))
    )

    rng = random.Random(config.seed)
    contexts: list[PolicyContext] = []
    for _ in range(config.pool_size):
        total_blocks = int(_log_uniform(rng, *BLOCKS_RANGE))
        hit_ratio = rng.uniform(*HIT_RATIO_RANGE)
        miss_blocks = int(round(total_blocks * (1.0 - hit_ratio)))
        contexts.append(
            PolicyContext(
                total_blocks=total_blocks,
                miss_blocks=min(total_blocks, max(0, miss_blocks)),
                storage_throughput=storage_throughput,
                recompute_throughput=recompute_throughput,
                request_rate=_log_uniform(rng, *REQUEST_RATE_RANGE),
                p95_seconds=rng.uniform(*P95_RANGE),
                stack_key=hardware.stack_key,
            )
        )
    return contexts


def _profile_policy(
    policy: PolicyBase,
    contexts: list[PolicyContext],
    is_baseline: bool,
) -> tuple[PolicyProfile, PoolTrace]:
    """Run the policy over the pool once to get its service time and branch mix."""
    service_times: list[float] = []
    modes: list[str] = []
    counts: dict[str, int] = {}
    for ctx in contexts:
        result = evaluate(policy, ctx)
        service_times.append(result.total_time)
        modes.append(result.decision_mode)
        counts[result.decision_mode] = counts.get(result.decision_mode, 0) + 1

    finite = [value for value in service_times if value != float("inf")]
    profile = PolicyProfile(
        name=policy.name,
        label=policy.label,
        color=policy.color,
        mean_service_seconds=statistics.fmean(finite) if finite else float("inf"),
        decision_modes=counts,
        is_baseline=is_baseline,
    )
    return profile, PoolTrace(service_seconds=service_times, decision_modes=modes)


def _time_combinations(
    policy: PolicyBase,
    contexts: list[PolicyContext],
    reps: int,
    rounds: int,
) -> list[float]:
    """Per-decision microseconds for *each* workload combination in the pool.

    One number per combination: the median over ``rounds`` of a tight ``reps``
    loop. The loop's own interpreter overhead is inside every measurement, which
    is exactly why the summary reports overhead *relative to* the no-logic
    baseline policies - that subtraction cancels it out.
    """
    decide = policy.decide
    costs: list[float] = []
    for ctx in contexts:
        samples: list[float] = []
        for _ in range(rounds):
            start = time.perf_counter()
            for _ in range(reps):
                decide(ctx)
            samples.append(time.perf_counter() - start)
        costs.append(1e6 * statistics.median(samples) / reps)
    return costs


def _trials_for(request_count: int) -> int:
    for threshold, trials in TRIAL_SCHEDULE:
        if threshold is None or request_count <= threshold:
            return trials
    return TRIAL_SCHEDULE[-1][1]


def _time_decisions(
    policy: PolicyBase,
    contexts: list[PolicyContext],
    request_count: int,
) -> float:
    """Median wall-clock seconds to decide ``request_count`` requests.

    Each trial resumes the stream where the previous one stopped, so the micro
    points (R=1, R=3) sample across the whole pool instead of replaying the same
    first request and reporting whatever branch it happens to take. Rotation is
    done outside the timer; inside it, ``cycle``/``islice`` keep stream
    generation in C so the loop is dominated by ``decide()``.
    """
    decide = policy.decide
    pool_size = len(contexts)
    samples: list[float] = []
    for trial in range(_trials_for(request_count)):
        offset = (trial * request_count) % pool_size
        rotated = contexts[offset:] + contexts[:offset]
        stream = islice(cycle(rotated), request_count)
        start = time.perf_counter()
        for ctx in stream:
            decide(ctx)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def _warmup(policies: list[PolicyBase], contexts: list[PolicyContext], count: int) -> None:
    """Touch every code path before timing, so no policy pays first-call costs."""
    for policy in policies:
        for ctx in islice(cycle(contexts), count):
            policy.decide(ctx)


@dataclass(frozen=True)
class SweepOutcome:
    """Everything one run produces, before it is written out."""

    profiles: dict[str, PolicyProfile]
    series: dict[str, list[dict[str, Any]]]
    combination_costs: dict[str, list[float]]
    summary: dict[str, Any]


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; ``statistics.quantiles`` needs n >= 2 points."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def _describe(values: list[float]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(values) if values else float("nan"),
        "median_us": statistics.median(values) if values else float("nan"),
        "p95_us": _percentile(values, 0.95),
        "min_us": min(values) if values else float("nan"),
        "max_us": max(values) if values else float("nan"),
        "stdev_us": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _baseline_costs(
    combination_costs: dict[str, list[float]],
    baseline_names: list[str],
) -> list[float]:
    """The no-logic floor per combination: mean cost across the baseline policies.

    Restore and Recompute both return a constant ``k``, so whatever they cost is
    dispatch plus loop overhead, not decision-making. Averaging them gives a
    steadier floor than picking either one.
    """
    columns = [combination_costs[name] for name in baseline_names]
    return [statistics.fmean(row) for row in zip(*columns)]


def _summarize(
    profiles: dict[str, PolicyProfile],
    traces: dict[str, PoolTrace],
    combination_costs: dict[str, list[float]],
    series: dict[str, list[dict[str, Any]]],
    config: OverheadConfig,
) -> dict[str, Any]:
    """Aggregate every timed combination into one quotable table per policy."""
    baseline_names = [
        name for name in config.baseline_policies if name in combination_costs
    ]
    baseline = _baseline_costs(combination_costs, baseline_names) if baseline_names else []

    combinations = config.pool_size
    per_policy_decisions = combinations * config.combination_reps * config.combination_rounds
    swept_decisions = sum(
        point["request_count"] * point["trials"]
        for points in series.values()
        for point in points
    )

    policies: dict[str, Any] = {}
    for name, costs in combination_costs.items():
        trace = traces[name]
        stats = _describe(costs)

        # Overhead each combination pays over the no-logic floor, and what share
        # of that request's modeled service time it represents.
        added = [cost - floor for cost, floor in zip(costs, baseline)] if baseline else []
        relative = [
            100.0 * (delta * 1e-6) / service
            for delta, service in zip(added, trace.service_seconds)
            if service > 0 and service != float("inf")
        ]

        by_mode: dict[str, Any] = {}
        for mode in sorted(set(trace.decision_modes)):
            mode_costs = [
                cost
                for cost, ctx_mode in zip(costs, trace.decision_modes)
                if ctx_mode == mode
            ]
            by_mode[mode] = {"combinations": len(mode_costs), **_describe(mode_costs)}

        baseline_mean = statistics.fmean(baseline) if baseline else float("nan")
        policies[name] = {
            "label": profiles[name].label,
            "is_baseline": profiles[name].is_baseline,
            "combinations": combinations,
            "decisions_measured": per_policy_decisions,
            "cost": stats,
            "added_overhead": {
                "mean_us": statistics.fmean(added) if added else 0.0,
                "median_us": statistics.median(added) if added else 0.0,
                "p95_us": _percentile(added, 0.95) if added else 0.0,
                "max_us": max(added) if added else 0.0,
                "times_baseline": (
                    stats["mean_us"] / baseline_mean if baseline_mean > 0 else float("nan")
                ),
                "mean_percent_of_request_time": (
                    statistics.fmean(relative) if relative else 0.0
                ),
                "median_percent_of_request_time": (
                    statistics.median(relative) if relative else 0.0
                ),
            },
            "by_decision_mode": by_mode,
            "mean_service_seconds": profiles[name].mean_service_seconds,
            "decisions_per_second": (
                1e6 / stats["mean_us"] if stats["mean_us"] > 0 else float("inf")
            ),
        }

    return {
        "combinations": combinations,
        "reps_per_combination": config.combination_reps,
        "rounds_per_combination": config.combination_rounds,
        "decisions_measured_per_policy": per_policy_decisions,
        "decisions_measured_in_sweep": swept_decisions,
        "baseline_policies": baseline_names,
        "baseline_cost": _describe(baseline) if baseline else {},
        "policies": policies,
    }


def _run_sweep(hardware: HardwareConfig, config: OverheadConfig) -> SweepOutcome:
    policies = get_policies()
    contexts = _build_context_pool(hardware, config)

    profiles: dict[str, PolicyProfile] = {}
    traces: dict[str, PoolTrace] = {}
    for policy in policies:
        profile, trace = _profile_policy(
            policy, contexts, policy.name in config.baseline_policies
        )
        profiles[policy.name] = profile
        traces[policy.name] = trace

    _warmup(policies, contexts, config.warmup_decisions)

    series: dict[str, list[dict[str, Any]]] = {policy.name: [] for policy in policies}
    for request_count in config.request_counts:
        for policy in policies:
            total_seconds = _time_decisions(policy, contexts, request_count)
            per_decision_us = 1e6 * total_seconds / request_count
            service_seconds = profiles[policy.name].mean_service_seconds
            relative_percent = (
                100.0 * (total_seconds / request_count) / service_seconds
                if service_seconds > 0 and service_seconds != float("inf")
                else 0.0
            )
            series[policy.name].append(
                {
                    "request_count": request_count,
                    "trials": _trials_for(request_count),
                    "total_seconds": total_seconds,
                    "per_decision_us": per_decision_us,
                    "decisions_per_second": (
                        request_count / total_seconds if total_seconds > 0 else float("inf")
                    ),
                    "relative_overhead_percent": relative_percent,
                }
            )
        print(f"  swept R={request_count:,}")

    print(f"  timing {config.pool_size:,} workload combinations individually ...")
    combination_costs = {
        policy.name: _time_combinations(
            policy, contexts, config.combination_reps, config.combination_rounds
        )
        for policy in policies
    }

    summary = _summarize(profiles, traces, combination_costs, series, config)
    return SweepOutcome(
        profiles=profiles,
        series=series,
        combination_costs=combination_costs,
        summary=summary,
    )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": FONT_SIZE,
            "axes.labelsize": AXIS_LABEL_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "axes.linewidth": 0.8,
        }
    )


def _compact_tick(value: float, _position: int) -> str:
    """Readable log tick: 0.2, 5, 1e-05 - never matplotlib's '6 x 10^0'."""
    return f"{value:g}"


def _style_log_axis(axis) -> None:
    """A 1-2-5 ladder per decade, so a two-decade range still has usable ticks."""
    axis.set_major_locator(ticker.LogLocator(base=10.0))
    axis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(2.0, 5.0)))
    axis.set_major_formatter(ticker.FuncFormatter(_compact_tick))
    axis.set_minor_formatter(ticker.FuncFormatter(_compact_tick))


def _build_line_figure(
    profiles: dict[str, PolicyProfile],
    series: dict[str, list[dict[str, Any]]],
    metric: str,
    y_label: str,
    log_y: bool,
    output_figure: Path,
) -> None:
    _apply_style()

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BACKGROUND_COLOR)

    for name, points in series.items():
        profile = profiles[name]
        ax.plot(
            [point["request_count"] for point in points],
            [point[metric] for point in points],
            label=profile.label,
            color=profile.color,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
        )

    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
        _style_log_axis(ax.yaxis)
        ax.tick_params(axis="y", which="minor", labelsize=TICK_LABEL_SIZE - 2)
    ax.set_xlabel("Requests Scheduled")
    ax.set_ylabel(y_label)
    ax.grid(which="major", color=GRID_COLOR, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="upper center",
        ncol=min(3, len(series)),
        bbox_to_anchor=(0.44, 1.26 if len(series) <= 3 else 1.40),
    )

    fig.tight_layout()
    fig.savefig(output_figure, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _build_distribution_figure(
    profiles: dict[str, PolicyProfile],
    combination_costs: dict[str, list[float]],
    output_figure: Path,
) -> None:
    """Spread of per-decision cost across every workload combination."""
    _apply_style()

    names = list(combination_costs.keys())
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BACKGROUND_COLOR)

    boxes = ax.boxplot(
        [combination_costs[name] for name in names],
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        # Mean *and* median: the IO-aware policy is bimodal (balanced vs
        # best-effort branch), so the two land far apart and that is the point.
        showmeans=True,
        meanprops={"marker": "D", "markersize": 3.5, "markerfacecolor": "#ffffff",
                   "markeredgecolor": "#1a1a1a", "markeredgewidth": 0.8},
        flierprops={"marker": ".", "markersize": 2, "markerfacecolor": "#555555",
                    "markeredgecolor": "none"},
        medianprops={"color": "#1a1a1a", "linewidth": 1.2},
        whiskerprops={"color": "#1a1a1a", "linewidth": 0.8},
        capprops={"color": "#1a1a1a", "linewidth": 0.8},
    )
    for patch, name in zip(boxes["boxes"], names):
        patch.set_facecolor(profiles[name].color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#1a1a1a")
        patch.set_linewidth(0.8)

    ax.set_yscale("log")
    _style_log_axis(ax.yaxis)
    ax.tick_params(axis="y", which="minor", labelsize=TICK_LABEL_SIZE - 2)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([profiles[name].label for name in names], rotation=12)
    ax.set_ylabel("Time per Decision (µs)")
    ax.grid(which="major", axis="y", color=GRID_COLOR, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_figure, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _build_hardware(args: argparse.Namespace) -> HardwareConfig:
    return HardwareConfig(
        stack_key=args.stack,
        gpu_key=args.gpu,
        dram_key=args.dram,
        disk_key=args.disk,
        link_key=args.link,
        gpu_count=args.gpu_count,
        dram_count=args.dram_count,
    )


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return tuple(int(value) for value in values)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure per-policy scheduler overhead from micro to macro request counts."
    )
    parser.add_argument("--output-figure-prefix", default=OUTPUT_FIGURE)
    parser.add_argument("--output-results", default=OUTPUT_RESULTS)
    parser.add_argument("--stack", default=STACK_KEY)
    parser.add_argument("--gpu", choices=tuple(GPUS.keys()), default=GPU_KEY)
    parser.add_argument("--dram", choices=tuple(DRAMS.keys()), default=DRAM_KEY)
    parser.add_argument("--disk", choices=tuple(DISKS.keys()), default=DISK_KEY)
    parser.add_argument("--link", choices=tuple(LINKS.keys()), default=LINK_KEY)
    parser.add_argument("--gpu-count", type=int, default=GPU_COUNT)
    parser.add_argument("--dram-count", type=int, default=DRAM_COUNT)
    parser.add_argument(
        "--request-counts",
        default=",".join(str(value) for value in REQUEST_COUNTS),
        help="Comma-separated request-stream sizes to sweep (micro to macro).",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Drop sweep points above this request count (0 keeps them all).",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=POOL_SIZE,
        help="Number of distinct workload combinations in the pool.",
    )
    parser.add_argument("--seed", type=int, default=POOL_SEED)
    parser.add_argument("--warmup-decisions", type=int, default=WARMUP_DECISIONS)
    parser.add_argument(
        "--combination-reps",
        type=int,
        default=COMBINATION_REPS,
        help="Decisions timed per workload combination, per round.",
    )
    parser.add_argument(
        "--combination-rounds",
        type=int,
        default=COMBINATION_ROUNDS,
        help="Rounds per combination; the median round is kept.",
    )
    parser.add_argument(
        "--baseline-policies",
        default=",".join(BASELINE_POLICIES),
        help="Comma-separated no-logic policies that define the overhead floor.",
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--font-size", type=int, default=FONT_SIZE)
    return parser


def _print_summary(outcome: SweepOutcome) -> None:
    """The quotable table: what each policy costs, over a do-nothing baseline."""
    summary = outcome.summary
    baseline_labels = ", ".join(
        outcome.profiles[name].label for name in summary["baseline_policies"]
    )

    print(
        f"\nPer-combination summary: {summary['combinations']:,} workload combinations"
        f" x {summary['reps_per_combination']:,} reps"
        f" x {summary['rounds_per_combination']} rounds"
        f" = {summary['decisions_measured_per_policy']:,} timed decisions per policy"
    )
    if baseline_labels:
        print(f"Baseline (no decision logic): {baseline_labels}")

    header = (
        f"  {'Policy':<18}{'median':>9}{'mean':>9}{'p95':>9}{'max':>9}"
        f"{'+median':>10}{'+mean':>9}{'x base':>9}{'% req':>11}"
    )
    print(f"\n{header}")
    print(f"  {'':-<18}{'':->9}{'':->9}{'':->9}{'':->9}{'':->10}{'':->9}{'':->9}{'':->11}")
    for name, entry in summary["policies"].items():
        cost = entry["cost"]
        added = entry["added_overhead"]
        marker = " *" if entry["is_baseline"] else ""
        print(
            f"  {entry['label'] + marker:<18}"
            f"{cost['median_us']:>9.3f}{cost['mean_us']:>9.3f}"
            f"{cost['p95_us']:>9.3f}{cost['max_us']:>9.3f}"
            f"{added['median_us']:>10.3f}{added['mean_us']:>9.3f}"
            f"{added['times_baseline']:>8.1f}x"
            f"{added['mean_percent_of_request_time']:>11.2e}"
        )
    print("  microseconds per decision; '+' columns are over baseline; * = baseline")

    for name, entry in summary["policies"].items():
        if entry["is_baseline"]:
            continue
        added = entry["added_overhead"]
        print(
            f"\n  {entry['label']}: across {entry['combinations']:,} workload combinations"
            f" ({entry['decisions_measured']:,} timed decisions), it added"
            f" {added['mean_us']:.2f} us mean / {added['median_us']:.2f} us median"
            f" per decision over the no-logic baseline ({added['times_baseline']:.1f}x),"
            f" i.e. {added['mean_percent_of_request_time']:.1e} % of modeled request time."
        )
        modes = entry["by_decision_mode"]
        if len(modes) > 1:
            detail = ", ".join(
                f"{mode} {stats['combinations']}/{entry['combinations']}"
                f" at {stats['median_us']:.2f} us"
                for mode, stats in modes.items()
            )
            print(f"    by branch: {detail}")


def main(argv: list[str] | None = None) -> int:
    global DPI, FONT_SIZE, AXIS_LABEL_SIZE, TICK_LABEL_SIZE

    parser = _build_parser()
    args = parser.parse_args(argv)

    DPI = args.dpi
    FONT_SIZE = args.font_size
    AXIS_LABEL_SIZE = args.font_size
    TICK_LABEL_SIZE = args.font_size

    request_counts = _parse_int_csv(args.request_counts)
    if any(value <= 0 for value in request_counts):
        raise ValueError("--request-counts must use positive values")
    if args.max_requests > 0:
        request_counts = tuple(
            value for value in request_counts if value <= args.max_requests
        )
        if not request_counts:
            raise ValueError("--max-requests removed every sweep point")
    if args.pool_size <= 0:
        raise ValueError("--pool-size must be greater than zero")
    if args.combination_reps <= 0 or args.combination_rounds <= 0:
        raise ValueError("--combination-reps / --combination-rounds must be positive")

    hardware = _build_hardware(args)
    config = OverheadConfig(
        request_counts=request_counts,
        pool_size=args.pool_size,
        seed=args.seed,
        warmup_decisions=max(0, args.warmup_decisions),
        combination_reps=args.combination_reps,
        combination_rounds=args.combination_rounds,
        baseline_policies=tuple(
            part.strip() for part in args.baseline_policies.split(",") if part.strip()
        ),
    )

    print(f"Measuring scheduler overhead on {hardware.stack_key} ...")
    outcome = _run_sweep(hardware, config)

    payload = {
        "hardware": asdict(hardware),
        "sweep": asdict(config),
        "workload_pool": {
            "blocks_range": list(BLOCKS_RANGE),
            "hit_ratio_range": list(HIT_RATIO_RANGE),
            "request_rate_range": list(REQUEST_RATE_RANGE),
            "p95_range": list(P95_RANGE),
        },
        "policies": {
            name: asdict(profile) for name, profile in outcome.profiles.items()
        },
        "summary": outcome.summary,
        "series": outcome.series,
        "combination_costs_us": outcome.combination_costs,
    }

    output_results = Path(args.output_results)
    output_results.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_prefix = Path(args.output_figure_prefix)
    suffix = output_prefix.suffix or ".pdf"
    figures = (
        ("total_seconds", "Scheduler Time (s)", True, "total"),
        ("per_decision_us", "Time per Decision (µs)", True, "per_request"),
        ("relative_overhead_percent", "Overhead (% of Request Time)", True, "relative"),
    )
    for metric, y_label, log_y, stem in figures:
        output_figure = output_prefix.with_name(f"{output_prefix.stem}_{stem}{suffix}")
        _build_line_figure(outcome.profiles, outcome.series, metric, y_label, log_y, output_figure)
        print(f"Saved figure -> {output_figure}")

    distribution_figure = output_prefix.with_name(
        f"{output_prefix.stem}_distribution{suffix}"
    )
    _build_distribution_figure(
        outcome.profiles, outcome.combination_costs, distribution_figure
    )
    print(f"Saved figure -> {distribution_figure}")
    print(f"Saved results -> {output_results}")

    _print_summary(outcome)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
