from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from simulator import ExperimentConfig, HardwareConfig, run_experiments
from src.pool import DISKS, DRAMS, GPUS, LINKS

# Default sweep settings. Edit these values or override them via the CLI.
STACK_KEY = "a5000"
GPU_KEY = "A5000"
DRAM_KEY = "DDR5"
DISK_KEY = "NVMe"
LINK_KEY = "NVLink"
GPU_COUNT = 1
DRAM_COUNT = 4
TOTAL_BLOCKS = 10_000
HIT_RATIO = 50
P95_VALUES = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
REQUEST_RATES = tuple(round(value, 2) for value in np.arange(0.01, 0.301, 0.01))
CACHE_RATIO_SWEEP = tuple(range(0, 101, 10))

# Plot controls.
OUTPUT_FIGURE = "a5000_sweep_map"
OUTPUT_RESULTS = "a5000_sweep_results.json"
FIG_WIDTH_PER_COL = 0.1
FIG_HEIGHT_PER_ROW = 0.1
FIG_EXTRA_WIDTH = 2.2
FIG_EXTRA_HEIGHT = 1.8
DPI = 300
FONT_SIZE = 10
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 10
LEGEND_FONT_SIZE = 8
EDGE_COLOR = "#1a1a1a"
EDGE_WIDTH = 0.7
BACKGROUND_COLOR = "#ffffff"
COLOR_FAIL_BOTH = (0.88, 0.11, 0.11)
COLOR_PASS_RPC = (0.98, 0.70, 0.20)
COLOR_PASS_LATENCY = (0.40, 0.65, 0.95)
COLOR_FULL_PATH = (0.20, 0.75, 0.35)
POLICY_LINE_COLORS = {
    "default_policy": "#d62728",
    "all_compute": "#2ca02c",
    "performance_aware": "#1f77b4",
}

POLICY_PLOTS = (
    ("default_policy", "Restoration"),
    ("all_compute", "Recomputation"),
    ("performance_aware", "SPA"),
)
POLICY_KEYS = tuple(name for name, _ in POLICY_PLOTS)


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return tuple(int(value) for value in values)


def _parse_float_csv(raw: str) -> tuple[float, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return tuple(float(value) for value in values)


def _validate_inputs() -> None:
    if TOTAL_BLOCKS <= 0:
        raise ValueError("--total-blocks must be greater than zero")
    if not P95_VALUES:
        raise ValueError("--p95-values must contain at least one value")
    if not REQUEST_RATES:
        raise ValueError("--request-rates must contain at least one value")
    if any(p95_value <= 0 for p95_value in P95_VALUES):
        raise ValueError("--p95-values must use positive values")
    if any(request_rate <= 0 for request_rate in REQUEST_RATES):
        raise ValueError("--request-rates must use positive values")
    if HIT_RATIO < 0 or HIT_RATIO > 100:
        raise ValueError("--hit-ratio must be a percentage in [0, 100]")
    if any(value < 0 or value > 100 for value in CACHE_RATIO_SWEEP):
        raise ValueError("cache ratio sweep must stay within [0, 100]")
    if GPU_KEY not in GPUS:
        raise ValueError(f"Unknown GPU key: {GPU_KEY}")
    if DRAM_KEY not in DRAMS:
        raise ValueError(f"Unknown DRAM key: {DRAM_KEY}")
    if DISK_KEY not in DISKS:
        raise ValueError(f"Unknown disk key: {DISK_KEY}")
    if LINK_KEY not in LINKS:
        raise ValueError(f"Unknown link key: {LINK_KEY}")
    if GPU_COUNT <= 0 or DRAM_COUNT <= 0:
        raise ValueError("--gpu-count and --dram-count must be greater than zero")


def _policy_state(result_row: dict[str, Any]) -> tuple[bool, bool]:
    rpc_pass = result_row["gpu_violation"] == 0.0 and result_row["storage_violation"] == 0.0
    latency_pass = result_row["latency_violation"] == 0.0
    return rpc_pass, latency_pass


def _state_rgb(state: tuple[bool, bool]) -> tuple[float, float, float]:
    if state == (False, False):
        return COLOR_FAIL_BOTH
    if state == (True, False):
        return COLOR_PASS_RPC
    if state == (False, True):
        return COLOR_PASS_LATENCY
    return COLOR_FULL_PATH


def _build_hardware() -> HardwareConfig:
    return HardwareConfig(
        stack_key=STACK_KEY,
        gpu_key=GPU_KEY,
        dram_key=DRAM_KEY,
        disk_key=DISK_KEY,
        link_key=LINK_KEY,
        gpu_count=GPU_COUNT,
        dram_count=DRAM_COUNT,
    )


def _collect_results() -> tuple[
    list[dict[str, Any]],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    hardware = _build_hardware()
    result_rows: list[dict[str, Any]] = []

    color_grids: dict[str, np.ndarray] = {
        policy_key: np.zeros((len(P95_VALUES), len(REQUEST_RATES), 3), dtype=float)
        for policy_key in POLICY_KEYS
    }
    state_grids: dict[str, np.ndarray] = {
        policy_key: np.empty((len(P95_VALUES), len(REQUEST_RATES)), dtype=object)
        for policy_key in POLICY_KEYS
    }

    miss_blocks = int(round(TOTAL_BLOCKS * (100 - HIT_RATIO) / 100))
    hit_blocks = TOTAL_BLOCKS - miss_blocks

    for p95_index, p95_value in enumerate(P95_VALUES):
        for rate_index, request_rate in enumerate(REQUEST_RATES):
            experiment = ExperimentConfig(
                total_blocks=(TOTAL_BLOCKS,),
                miss_blocks=(miss_blocks,),
                request_rate=request_rate,
                p95_seconds=float(p95_value),
            )
            row = run_experiments(hardware, experiment)[0]

            result_rows.append(
                {
                    "hit_ratio": HIT_RATIO,
                    "p95_seconds": p95_value,
                    "request_rate": request_rate,
                    "effective_slo_seconds": row["slo_seconds"],
                    "total_blocks": TOTAL_BLOCKS,
                    "hit_blocks": hit_blocks,
                    "miss_blocks": miss_blocks,
                    "default_policy_pass": row["default_policy"]["meets_slo"],
                    "all_compute_pass": row["all_compute"]["meets_slo"],
                    "performance_aware_pass": row["performance_aware"]["meets_slo"],
                    "performance_aware": row["performance_aware"],
                    "default_policy": row["default_policy"],
                    "all_compute": row["all_compute"],
                    "speedup_default_vs_all_compute": row[
                        "speedup_default_vs_all_compute"
                    ],
                    "speedup_all_compute_vs_performance_aware": row[
                        "speedup_all_compute_vs_performance_aware"
                    ],
                    "speedup_performance_aware_vs_default": row[
                        "speedup_performance_aware_vs_default"
                    ],
                }
            )

            for policy_key in POLICY_KEYS:
                policy_row = row[policy_key]
                state = _policy_state(policy_row)
                color_grids[policy_key][p95_index, rate_index, :] = _state_rgb(state)
                state_grids[policy_key][p95_index, rate_index] = state

    return result_rows, color_grids, state_grids


def _collect_success_rates() -> dict[str, list[float]]:
    hardware = _build_hardware()
    total_cases = len(P95_VALUES) * len(REQUEST_RATES)
    success_rates: dict[str, list[float]] = {policy_key: [] for policy_key in POLICY_KEYS}

    for cache_ratio in CACHE_RATIO_SWEEP:
        miss_blocks = int(round(TOTAL_BLOCKS * (100 - cache_ratio) / 100))
        counts = {policy_key: 0 for policy_key in POLICY_KEYS}

        for p95_value in P95_VALUES:
            for request_rate in REQUEST_RATES:
                experiment = ExperimentConfig(
                    total_blocks=(TOTAL_BLOCKS,),
                    miss_blocks=(miss_blocks,),
                    request_rate=request_rate,
                    p95_seconds=float(p95_value),
                )
                row = run_experiments(hardware, experiment)[0]
                for policy_key in POLICY_KEYS:
                    counts[policy_key] += int(bool(row[policy_key]["meets_slo"]))

        for policy_key in POLICY_KEYS:
            success_rates[policy_key].append(100.0 * counts[policy_key] / total_cases)

    return success_rates


def _build_figure(
    policy_key: str,
    policy_label: str,
    color_grid: np.ndarray,
    state_grid: np.ndarray,
    output_figure: Path,
) -> None:
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

    fig_width = FIG_WIDTH_PER_COL * len(REQUEST_RATES) + FIG_EXTRA_WIDTH
    fig_height = FIG_HEIGHT_PER_ROW * len(P95_VALUES) + FIG_EXTRA_HEIGHT
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.imshow(color_grid, origin="lower", interpolation="nearest")

    x_major_ticks = [0, len(REQUEST_RATES) // 2, len(REQUEST_RATES) - 1]
    y_major_ticks = [0, len(P95_VALUES) // 2, len(P95_VALUES) - 1]
    ax.set_xticks(x_major_ticks)
    ax.set_xticklabels([f"{REQUEST_RATES[index]:g}" for index in x_major_ticks])
    ax.set_yticks(y_major_ticks)
    ax.set_yticklabels([str(P95_VALUES[index]) for index in y_major_ticks])
    ax.set_xlabel("Request Rate (req/s)")
    ax.set_ylabel("P95 Latency (s)")
    ax.set_title(
        f"Hit Ratio={HIT_RATIO}%",
        fontsize=AXIS_LABEL_SIZE,
    )

    ax.set_xticks(np.arange(-0.5, len(REQUEST_RATES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(P95_VALUES), 1), minor=True)
    ax.grid(which="minor", color=EDGE_COLOR, linestyle="-", linewidth=EDGE_WIDTH)
    ax.tick_params(which="minor", bottom=False, left=False)

    channel_handles = [
        Patch(facecolor=COLOR_FAIL_BOTH, edgecolor=EDGE_COLOR, label="Fail"),
        Patch(facecolor=COLOR_PASS_RPC, edgecolor=EDGE_COLOR, label="RPC Only"),
        Patch(
            facecolor=COLOR_PASS_LATENCY,
            edgecolor=EDGE_COLOR,
            label="P95 Only",
        ),
        Patch(facecolor=COLOR_FULL_PATH, edgecolor=EDGE_COLOR, label="Pass"),
    ]

    fig.legend(
        handles=channel_handles,
        loc="upper center",
        bbox_to_anchor=(0.44, 0.9),
        frameon=False,
        ncol=4,
        fontsize=LEGEND_FONT_SIZE,
    )

    fig.subplots_adjust(right=0.80)
    fig.tight_layout(rect=(0.0, 0.0, 0.80, 1.0))
    fig.savefig(output_figure, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _build_success_rate_figure(
    success_rates: dict[str, list[float]],
    output_figure: Path,
) -> None:
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

    fig, ax = plt.subplots(figsize=(4.0, 2.2), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BACKGROUND_COLOR)

    for policy_key, policy_label in POLICY_PLOTS:
        ax.plot(
            CACHE_RATIO_SWEEP,
            success_rates[policy_key],
            label=policy_label,
            color=POLICY_LINE_COLORS[policy_key],
            linewidth=2.0,
            marker="o",
            markersize=4,
        )

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("Cache Ratio (%)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    # ax.grid(which="major", color=EDGE_COLOR, linestyle="-", linewidth=EDGE_WIDTH)
    # ax.set_title(
    #     f"Policy Success Rate Across r/p Sweep (N={TOTAL_BLOCKS:,})",
    #     fontsize=AXIS_LABEL_SIZE,
    # )
    ax.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.44, 1.28))

    fig.tight_layout()
    fig.savefig(output_figure, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep P95 SLO values and request rates using the KV cache simulator."
    )
    parser.add_argument(
        "--output-figure-prefix",
        default=OUTPUT_FIGURE,
        help="Output path prefix for the plot images. One file per policy is written.",
    )
    parser.add_argument("--output-results", default=OUTPUT_RESULTS)
    parser.add_argument("--stack", default=STACK_KEY)
    parser.add_argument("--gpu", choices=tuple(GPUS.keys()), default=GPU_KEY)
    parser.add_argument("--dram", choices=tuple(DRAMS.keys()), default=DRAM_KEY)
    parser.add_argument("--disk", choices=tuple(DISKS.keys()), default=DISK_KEY)
    parser.add_argument("--link", choices=tuple(LINKS.keys()), default=LINK_KEY)
    parser.add_argument("--gpu-count", type=int, default=GPU_COUNT)
    parser.add_argument("--dram-count", type=int, default=DRAM_COUNT)
    parser.add_argument("--total-blocks", type=int, default=TOTAL_BLOCKS)
    parser.add_argument("--hit-ratio", type=int, default=HIT_RATIO)
    parser.add_argument("--p95-values", default=",".join(map(str, P95_VALUES)))
    parser.add_argument("--slo-values", dest="p95_values", default=None)
    parser.add_argument(
        "--request-rates",
        default=",".join(f"{value:g}" for value in REQUEST_RATES),
    )
    parser.add_argument("--fig-width-per-col", type=float, default=FIG_WIDTH_PER_COL)
    parser.add_argument("--fig-height-per-row", type=float, default=FIG_HEIGHT_PER_ROW)
    parser.add_argument("--fig-extra-width", type=float, default=FIG_EXTRA_WIDTH)
    parser.add_argument("--fig-extra-height", type=float, default=FIG_EXTRA_HEIGHT)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--font-size", type=int, default=FONT_SIZE)
    parser.add_argument("--axis-label-size", type=int, default=AXIS_LABEL_SIZE)
    parser.add_argument("--tick-label-size", type=int, default=TICK_LABEL_SIZE)
    parser.add_argument("--legend-font-size", type=int, default=LEGEND_FONT_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    global STACK_KEY, GPU_KEY, DRAM_KEY, DISK_KEY, LINK_KEY, GPU_COUNT, DRAM_COUNT
    global TOTAL_BLOCKS, HIT_RATIO, P95_VALUES, REQUEST_RATES
    global FIG_WIDTH_PER_COL, FIG_HEIGHT_PER_ROW, FIG_EXTRA_WIDTH, FIG_EXTRA_HEIGHT
    global \
        DPI, \
        FONT_SIZE, \
        AXIS_LABEL_SIZE, \
        TICK_LABEL_SIZE, \
        LEGEND_FONT_SIZE

    parser = _build_parser()
    args = parser.parse_args(argv)

    STACK_KEY = args.stack
    GPU_KEY = args.gpu
    DRAM_KEY = args.dram
    DISK_KEY = args.disk
    LINK_KEY = args.link
    GPU_COUNT = args.gpu_count
    DRAM_COUNT = args.dram_count
    TOTAL_BLOCKS = args.total_blocks
    HIT_RATIO = args.hit_ratio
    P95_VALUES = _parse_int_csv(args.p95_values)
    REQUEST_RATES = _parse_float_csv(args.request_rates)
    FIG_WIDTH_PER_COL = args.fig_width_per_col
    FIG_HEIGHT_PER_ROW = args.fig_height_per_row
    FIG_EXTRA_WIDTH = args.fig_extra_width
    FIG_EXTRA_HEIGHT = args.fig_extra_height
    DPI = args.dpi
    FONT_SIZE = args.font_size
    AXIS_LABEL_SIZE = args.axis_label_size
    TICK_LABEL_SIZE = args.tick_label_size
    LEGEND_FONT_SIZE = args.legend_font_size

    _validate_inputs()

    results, color_grids, state_grids = _collect_results()
    success_rates = _collect_success_rates()

    payload = {
        "hardware": asdict(_build_hardware()),
        "total_blocks": TOTAL_BLOCKS,
        "hit_ratio": HIT_RATIO,
        "cache_ratio_sweep": list(CACHE_RATIO_SWEEP),
        "p95_values": list(P95_VALUES),
        "request_rates": list(REQUEST_RATES),
        "effective_slo_formula": "q = min(p95_seconds, 1 / request_rate)",
        "policy_success_rates": success_rates,
        "results": results,
    }

    output_results = Path(args.output_results)
    output_results.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_prefix = Path(args.output_figure_prefix)
    for policy_key, policy_label in POLICY_PLOTS:
        output_figure = output_prefix.with_name(
            f"{output_prefix.stem}_{policy_key}{output_prefix.suffix or '.png'}"
        )
        _build_figure(
            policy_key,
            policy_label,
            color_grids[policy_key],
            state_grids[policy_key],
            output_figure,
        )
        print(f"Saved figure -> {output_figure}")

    success_rate_figure = output_prefix.with_name(
        f"{output_prefix.stem}_success_rate{output_prefix.suffix or '.png'}"
    )

    _build_success_rate_figure(success_rates, success_rate_figure)

    print(f"Saved figure -> {success_rate_figure}")
    print(f"Saved results -> {output_results}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
