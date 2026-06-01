from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from simulator import ExperimentConfig, HardwareConfig, run_experiments

# Default sweep settings. Edit these values or override them via the CLI.
STACK_KEY = "h200"
TOTAL_BLOCKS = 50_000
HIT_RATIOS = (0, 25, 50, 75, 100)
SLO_VALUES = (10 , 20, 30, 40, 50, 60)
REQUEST_RATE = 0.04

# Plot controls.
OUTPUT_FIGURE = "h200_sweep_map.png"
OUTPUT_RESULTS = "h200_sweep_results.json"
FIG_WIDTH_PER_COL = 1.35
FIG_HEIGHT_PER_ROW = 0.90
FIG_EXTRA_WIDTH = 4.2
FIG_EXTRA_HEIGHT = 2.2
DPI = 300
FONT_SIZE = 11
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
ANNOTATION_SIZE = 8
LEGEND_FONT_SIZE = 9
EDGE_COLOR = "#1a1a1a"
EDGE_WIDTH = 0.7
BACKGROUND_COLOR = "#f2f2f2"

# RGB channels map to policies: red = default, green = all_compute, blue = performance_aware.
POLICY_CHANNELS = (
    ("default_policy", "Default Cache\nRestore Policy", (1.0, 0.0, 0.0)),
    ("all_compute", "All Compute", (0.0, 1.0, 0.0)),
    ("performance_aware", "Storage Performance\nAware Restore", (0.0, 0.0, 1.0)),
)
POLICY_KEYS = tuple(name for name, _, _ in POLICY_CHANNELS)


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return tuple(int(value) for value in values)


def _policy_state(result_row: dict) -> tuple[bool, bool, bool]:
    return tuple(result_row[name]["meets_slo"] for name in POLICY_KEYS)


def _state_label(state: tuple[bool, bool, bool]) -> str:
    label_map = {
        (False, False, False): "None",
        (True, False, False): "Rest.",
        (False, True, False): "Comp.",
        (False, False, True): "SPAP.",
        (True, True, False): "Rest.\nComp.",
        (True, False, True): "Rest.\nSPAP.",
        (False, True, True): "Comp.\nSPAP.",
        (True, True, True): "All",
    }
    return label_map[state]


def _state_rgb(state: tuple[bool, bool, bool]) -> tuple[float, float, float]:
    return tuple(1.0 if passed else 0.0 for passed in state)


def _state_name(state: tuple[bool, bool, bool]) -> str:
    return f"{int(state[0])}{int(state[1])}{int(state[2])}"


def _text_color(rgb: tuple[float, float, float]) -> str:
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return "white" if luminance < 0.55 else "black"


def _build_hardware() -> HardwareConfig:
    return HardwareConfig(
        stack_key=STACK_KEY,
        gpu_key="H200",
        dram_key="DDR5",
        disk_key="NVMe",
        link_key="NVLink",
    )


def _collect_results() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    hardware = _build_hardware()
    result_rows: list[dict] = []

    color_grid = np.zeros((len(HIT_RATIOS), len(SLO_VALUES), 3), dtype=float)
    state_grid = np.empty((len(HIT_RATIOS), len(SLO_VALUES)), dtype=object)

    for hit_index, hit_ratio in enumerate(HIT_RATIOS):
        miss_blocks = int(round(TOTAL_BLOCKS * (100 - hit_ratio) / 100))
        for slo_index, slo_value in enumerate(SLO_VALUES):
            experiment = ExperimentConfig(
                total_blocks=(TOTAL_BLOCKS,),
                miss_blocks=(miss_blocks,),
                request_rate=REQUEST_RATE,
                p95_seconds=float(slo_value),
            )
            row = run_experiments(hardware, experiment)[0]
            state = _policy_state(row)

            result_rows.append(
                {
                    "hit_ratio": hit_ratio,
                    "slo_value": slo_value,
                    "slo_seconds": row["slo_seconds"],
                    "miss_blocks": miss_blocks,
                    "state": _state_name(state),
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

            color_grid[hit_index, slo_index, :] = _state_rgb(state)
            state_grid[hit_index, slo_index] = state

    return result_rows, color_grid, state_grid


def _build_figure(
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

    fig_width = FIG_WIDTH_PER_COL * len(SLO_VALUES) + FIG_EXTRA_WIDTH
    fig_height = FIG_HEIGHT_PER_ROW * len(HIT_RATIOS) + FIG_EXTRA_HEIGHT
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor(BACKGROUND_COLOR)

    ax.imshow(color_grid, origin="lower", interpolation="nearest")

    ax.set_xticks(np.arange(len(SLO_VALUES)))
    ax.set_xticklabels([str(value) for value in SLO_VALUES])
    ax.set_yticks(np.arange(len(HIT_RATIOS)))
    ax.set_yticklabels([str(value) for value in HIT_RATIOS])
    ax.set_xlabel("SLO\nmin(p95, 1/rps)")
    ax.set_ylabel("Cache Hit Ratio (%)")
    # ax.set_title(f"H200 sweep: {TOTAL_BLOCKS:,} total blocks, request_rate={REQUEST_RATE}")

    ax.set_xticks(np.arange(-0.5, len(SLO_VALUES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(HIT_RATIOS), 1), minor=True)
    ax.grid(which="minor", color=EDGE_COLOR, linestyle="-", linewidth=EDGE_WIDTH)
    ax.tick_params(which="minor", bottom=False, left=False)

    for hit_index, hit_ratio in enumerate(HIT_RATIOS):
        for slo_index, slo_value in enumerate(SLO_VALUES):
            state = state_grid[hit_index, slo_index]
            rgb = tuple(color_grid[hit_index, slo_index, :])
            label = _state_label(state)
            ax.text(
                slo_index,
                hit_index,
                label,
                ha="center",
                va="center",
                color=_text_color(rgb),
                fontsize=ANNOTATION_SIZE,
                fontweight="bold",
            )

    channel_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=10,
            markerfacecolor=color,
            markeredgecolor=EDGE_COLOR,
            label=label,
        )
        for _, label, color in POLICY_CHANNELS
    ]

    state_handles = []
    legend_states = [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]
    for state in legend_states:
        state_handles.append(
            Patch(
                facecolor=_state_rgb(state),
                edgecolor=EDGE_COLOR,
                label=f"{_state_name(state)} = {_state_label(state)}",
            )
        )
    state_handles.append(
        Patch(
            facecolor="#d9d9d9",
            edgecolor=EDGE_COLOR,
            label="n/a = missing",
        )
    )

    fig.legend(
        handles=channel_handles,
        loc="upper left",
        bbox_to_anchor=(0.665, 0.95),
        frameon=False,
        title="RGB channels",
        title_fontsize=LEGEND_FONT_SIZE,
    )
    fig.legend(
        handles=state_handles,
        loc="lower left",
        bbox_to_anchor=(0.67, 0.40),
        frameon=False,
        title="Pass/fail states",
        title_fontsize=LEGEND_FONT_SIZE,
        ncol=1,
    )

    fig.subplots_adjust(right=0.76)
    fig.tight_layout(rect=(0.0, 0.0, 0.76, 1.0))
    fig.savefig(output_figure, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep H200 hit ratios and SLO values."
    )
    parser.add_argument("--output-figure", default=OUTPUT_FIGURE)
    parser.add_argument("--output-results", default=OUTPUT_RESULTS)
    parser.add_argument("--stack", default=STACK_KEY)
    parser.add_argument("--total-blocks", type=int, default=TOTAL_BLOCKS)
    parser.add_argument("--hit-ratios", default=",".join(map(str, HIT_RATIOS)))
    parser.add_argument("--slo-values", default=",".join(map(str, SLO_VALUES)))
    parser.add_argument("--request-rate", type=float, default=REQUEST_RATE)
    parser.add_argument("--fig-width-per-col", type=float, default=FIG_WIDTH_PER_COL)
    parser.add_argument("--fig-height-per-row", type=float, default=FIG_HEIGHT_PER_ROW)
    parser.add_argument("--fig-extra-width", type=float, default=FIG_EXTRA_WIDTH)
    parser.add_argument("--fig-extra-height", type=float, default=FIG_EXTRA_HEIGHT)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--font-size", type=int, default=FONT_SIZE)
    parser.add_argument("--axis-label-size", type=int, default=AXIS_LABEL_SIZE)
    parser.add_argument("--tick-label-size", type=int, default=TICK_LABEL_SIZE)
    parser.add_argument("--annotation-size", type=int, default=ANNOTATION_SIZE)
    parser.add_argument("--legend-font-size", type=int, default=LEGEND_FONT_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    global STACK_KEY, TOTAL_BLOCKS, HIT_RATIOS, SLO_VALUES, REQUEST_RATE
    global FIG_WIDTH_PER_COL, FIG_HEIGHT_PER_ROW, FIG_EXTRA_WIDTH, FIG_EXTRA_HEIGHT
    global \
        DPI, \
        FONT_SIZE, \
        AXIS_LABEL_SIZE, \
        TICK_LABEL_SIZE, \
        ANNOTATION_SIZE, \
        LEGEND_FONT_SIZE

    parser = _build_parser()
    args = parser.parse_args(argv)

    STACK_KEY = args.stack
    TOTAL_BLOCKS = args.total_blocks
    HIT_RATIOS = _parse_int_csv(args.hit_ratios)
    SLO_VALUES = _parse_int_csv(args.slo_values)
    REQUEST_RATE = args.request_rate
    FIG_WIDTH_PER_COL = args.fig_width_per_col
    FIG_HEIGHT_PER_ROW = args.fig_height_per_row
    FIG_EXTRA_WIDTH = args.fig_extra_width
    FIG_EXTRA_HEIGHT = args.fig_extra_height
    DPI = args.dpi
    FONT_SIZE = args.font_size
    AXIS_LABEL_SIZE = args.axis_label_size
    TICK_LABEL_SIZE = args.tick_label_size
    ANNOTATION_SIZE = args.annotation_size
    LEGEND_FONT_SIZE = args.legend_font_size

    results, color_grid, state_grid = _collect_results()

    payload = {
        "hardware": asdict(_build_hardware()),
        "total_blocks": TOTAL_BLOCKS,
        "hit_ratios": list(HIT_RATIOS),
        "slo_values": list(SLO_VALUES),
        "request_rate": REQUEST_RATE,
        "results": results,
    }

    output_results = Path(args.output_results)
    output_results.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    output_figure = Path(args.output_figure)
    _build_figure(color_grid, state_grid, output_figure)

    print(f"Saved figure -> {output_figure}")
    print(f"Saved results -> {output_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
