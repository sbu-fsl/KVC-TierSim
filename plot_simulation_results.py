import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


plt.rcParams["font.family"] = "serif"


before_color = "#0000bd"
after_color = "#ffa500"
grid_color = "#cccccc"
axis_color = "#bbbbbb"
text_color = "#444444"


def style_ax(ax, xlabels):
    ax.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(axis_color)
    ax.spines["bottom"].set_color(axis_color)
    ax.tick_params(colors=text_color, labelsize=8)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=8, rotation=0, ha="center", color=text_color)


def s_fmt(val, _):
    if val >= 3600:
        return f"{val / 3600:.0f}h"
    if val >= 60:
        return f"{val / 60:.0f}min"
    if val >= 1:
        return f"{val:.0f}s"
    return f"{val * 1000:.0f}ms"


def blocks_fmt(val, _):
    if val >= 1_000_000:
        return f"{val / 1_000_000:.0f}M"
    if val >= 1_000:
        return f"{val / 1_000:.0f}k"
    return f"{val:.0f}"


def pct_fmt(val, _):
    return f"{val:.0f}%"


def pretty_gpu_name(gpu_key):
    return {
        "A5000": "RTX A5000",
        "RTX6000": "RTX 6000 Ada",
    }.get(gpu_key, gpu_key)


def pretty_stack_name(stack_key, hardware):
    if not stack_key:
        return pretty_gpu_name(hardware.get("gpu_key", ""))
    return {
        "h200": "H200",
        "a5000": "RTX A5000",
        "a100_ddr5_nvme": "A100 DDR5/NVMe",
        "a100_ddr4_nvme": "A100 DDR4/NVMe",
        "a100_ddr4_sata": "A100 DDR4/SATA",
    }.get(stack_key, stack_key.replace("_", " "))


def format_blocks(blocks):
    return f"{blocks:,}"


def format_rate(rate):
    return f"{rate:g}"


def format_request_count(request_count):
    return f"{request_count:,}"


def build_case_label(case):
    return (
        f"{case['stack_label']}\n"
        f"N={format_blocks(case['total_blocks'])} / M={format_blocks(case['miss_blocks'])} / "
        f"r={format_rate(case['request_rate'])} / p={case['p95_seconds']:.0f}s"
    )


def scenario_axis_label(case):
    return (
        f"{case['stack_label']} / N={format_blocks(case['total_blocks'])} / M={format_blocks(case['miss_blocks'])} / "
        f"r={format_rate(case['request_rate'])} req/s / p={case['p95_seconds']:.0f}s"
    )


def load_cases(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    hardware = payload["hardware"]
    experiment = payload["experiment"]

    cases = []
    for result in payload["results"]:
        total_blocks = result.get("total_blocks", 0)
        miss_blocks = result.get("miss_blocks")
        if miss_blocks is None:
            miss_blocks = result["default_policy"].get("recompute_blocks", 0)

        stack_key = hardware.get("stack_key", "")
        stack_label = pretty_stack_name(stack_key, hardware)

        case = {
            "path": Path(path),
            "file_label": Path(path).stem,
            "stack_key": stack_key,
            "stack_label": stack_label,
            "gpu_key": hardware["gpu_key"],
            "gpu_label": pretty_gpu_name(hardware["gpu_key"]),
            "total_blocks": total_blocks,
            "miss_blocks": miss_blocks,
            "hit_blocks": result.get("hit_blocks", total_blocks - miss_blocks),
            "request_count": result.get("request_count", experiment.get("request_count", 1)),
            "request_rate": result.get("request_rate", experiment.get("request_rate", 1.0)),
            "p95_seconds": result.get("p95_seconds", experiment.get("p95_seconds", experiment.get("slo_seconds", 1.0))),
            "before_time": result["default_policy"]["total_time"],
            "after_time": result["performance_aware"]["total_time"],
            "before_storage_blocks": result["default_policy"]["storage_blocks"],
            "before_recompute_blocks": result["default_policy"]["recompute_blocks"],
            "after_storage_blocks": result["performance_aware"]["storage_blocks"],
            "after_recompute_blocks": result["performance_aware"]["recompute_blocks"],
            "before_success_rate": result["default_policy"].get("success_rate", 0.0),
            "after_success_rate": result["performance_aware"].get("success_rate", 0.0),
            "speedup": result["speedup"],
            "slo_seconds": result.get("slo_seconds", experiment.get("slo_seconds", 1.0)),
        }
        case["label"] = build_case_label(case)
        cases.append(case)

    return cases


def plot_latency_comparison(cases, output="policy_latency_comparison.pdf"):
    labels = [case["label"] for case in cases]
    x = np.arange(len(labels))
    width = min(0.36, 0.36 / max(1, len(cases) / 4))

    fig, ax = plt.subplots(figsize=(max(4.0, len(cases) * 1.2), 2.9))
    fig.patch.set_facecolor("#ffffff")
    style_ax(ax, labels)
    ax.set_xticklabels(labels, fontsize=8, rotation=0, ha="center", color=text_color)

    before = [case["before_time"] for case in cases]
    after = [case["after_time"] for case in cases]

    ax.bar(
        x - width / 2,
        before,
        width,
        label="Before policy",
        color=before_color,
        alpha=0.90,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        after,
        width,
        label="After policy",
        color=after_color,
        alpha=0.90,
        zorder=3,
    )

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(s_fmt))
    ax.set_yticks([0.1, 1, 10, 100, 1_000])
    ax.set_ylabel("Processing Time\n(log scale)", fontsize=10, color="#333")
    ax.set_xlabel(scenario_axis_label(cases[0]), fontsize=10, color="#333")
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.5, color=grid_color, zorder=0)
    ax.legend(
        facecolor="white",
        edgecolor="#ccc",
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        framealpha=0.95,
    )

    for idx, case in enumerate(cases):
        ax.text(
            x[idx],
            max(case["before_time"], case["after_time"]) * 1.08,
            f"{case['speedup']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())


def plot_policy_split(cases, output="policy_block_split.pdf"):
    labels = [case["label"] for case in cases]
    x = np.arange(len(labels))
    width = min(0.36, 0.36 / max(1, len(cases) / 4))

    fig, ax = plt.subplots(figsize=(max(4.0, len(cases) * 1.2), 2.9))
    fig.patch.set_facecolor("#ffffff")
    style_ax(ax, labels)
    ax.set_xticklabels(labels, fontsize=8, rotation=0, ha="center", color=text_color)

    default_storage = [case["before_storage_blocks"] for case in cases]
    default_recompute = [case["before_recompute_blocks"] for case in cases]
    aware_storage = [case["after_storage_blocks"] for case in cases]
    aware_recompute = [case["after_recompute_blocks"] for case in cases]

    ax.bar(
        x - width / 2,
        default_storage,
        width,
        label="Restoration",
        color=after_color,
        alpha=0.55,
        zorder=3,
    )
    ax.bar(
        x - width / 2,
        default_recompute,
        width,
        bottom=default_storage,
        label="Recomputation",
        color=before_color,
        alpha=0.55,
        zorder=3,
    )

    ax.bar(
        x + width / 2,
        aware_storage,
        width,
        label="Policy restoration",
        color=after_color,
        alpha=0.95,
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        aware_recompute,
        width,
        bottom=aware_storage,
        label="Policy recompute",
        color=before_color,
        alpha=0.95,
        zorder=3,
    )

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(blocks_fmt))
    ax.set_ylabel("KV Cache Blocks", fontsize=10, color="#333")
    ax.set_xlabel(
        scenario_axis_label(cases[0]),
        fontsize=10,
        color="#333",
    )
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.5, color=grid_color, zorder=0)
    ax.legend(
        facecolor="white",
        edgecolor="#ccc",
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=2,
        framealpha=0.95,
    )

    for idx, case in enumerate(cases):
        total = case["after_storage_blocks"] + case["after_recompute_blocks"]
        ax.text(
            x[idx],
            total * 1.03,
            f"SLO {case['slo_seconds']:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())


def plot_slo_rate(cases, output="policy_slo_rate.pdf"):
    labels = [case["label"] for case in cases]
    x = np.arange(len(labels))
    width = min(0.34, 0.34 / max(1, len(cases) / 4))

    fig, ax = plt.subplots(figsize=(max(4.0, len(cases) * 1.2), 2.9))
    fig.patch.set_facecolor("#ffffff")
    style_ax(ax, labels)
    ax.set_xticklabels(labels, fontsize=8, rotation=0, ha="center", color=text_color)

    old_rate = [case["before_success_rate"] * 100 for case in cases]
    new_rate = [case["after_success_rate"] * 100 for case in cases]
    ax.bar(x - width / 2, old_rate, width, label="Old policy", color=before_color, alpha=0.90, zorder=3)
    ax.bar(x + width / 2, new_rate, width, label="New policy", color=after_color, alpha=0.90, zorder=3)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(pct_fmt))
    ax.set_ylabel("Fraction of Requests\n> SLO(%)", fontsize=10, color="#333")
    ax.set_xlabel(scenario_axis_label(cases[0]), fontsize=10, color="#333")
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.5, color=grid_color, zorder=0)
    ax.legend(
        facecolor="white",
        edgecolor="#ccc",
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.17),
        ncol=2,
        framealpha=0.95,
    )

    for idx, case in enumerate(cases):
        ax.text(x[idx] - width / 2, old_rate[idx] + 2, f"{old_rate[idx]:.0f}%", ha="center", va="bottom", fontsize=8, color="#222")
        ax.text(x[idx] + width / 2, new_rate[idx] + 2, f"{new_rate[idx]:.0f}%", ha="center", va="bottom", fontsize=8, color="#222")

        # annotate the SLO information
        ax.text(
            x[idx],
            max(old_rate[idx], new_rate[idx]) + 10,
            f"SLO {case['slo_seconds']:.0f}s",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )

    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())


def build_parser():
    parser = argparse.ArgumentParser(description="Plot performance-aware KV cache simulation results.")
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["h200.json", "a5000.json"],
        help="JSON result files to plot.",
    )
    parser.add_argument("--latency-output", default="policy_latency_comparison.pdf")
    parser.add_argument("--split-output", default="policy_block_split.pdf")
    parser.add_argument("--slo-output", default="policy_slo_rate.pdf")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cases = []
    for path in args.inputs:
        cases.extend(load_cases(path))

    plot_latency_comparison(cases, output=args.latency_output)
    plot_policy_split(cases, output=args.split_output)
    plot_slo_rate(cases, output=args.slo_output)

    print(f"Saved {args.latency_output}, {args.split_output}, and {args.slo_output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())