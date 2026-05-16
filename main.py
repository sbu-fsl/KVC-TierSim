import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.lines import Line2D

from src.pool import DISKS, DRAMS, GPUS, LINKS

# set the global font family to serif
plt.rcParams["font.family"] = "serif"

# set the global axis line width
mpl.rcParams["axes.linewidth"] = 0.5

# Global configuration for the simulator
CONFIG = {
    "NPOINTS": 5000,         # Number of points to plot
    "TOKENS_PER_BLOCK": 16,  # Number of tokens per block
    "BYTES_PER_BLOCK": 1e6,  # 1 MB per block
    "MODEL_PARAMS": 8e9,     # 8 billion parameters
    # "BYTES_PER_BLOCK": 4e6,   # 4 MB per block
    # "MODEL_PARAMS": 32e9,     # 32 billion parameters
    # "BYTES_PER_BLOCK": 16e6,  # 16 MB per block
    # "MODEL_PARAMS": 128e9,    # 128 billion parameters
    "GPU_ETA": 0.5           # GPU effectiveness factor
}

# GPU family color map — same GPU key = same color
GPU_COLORS = {
    "H200":     "#e6194b",
    "H100":     "#f58231",
    "A100":     "#ffe119",
    "RTX6000":  "#3cb44b",
    "V100":     "#4363d8",
    "A5000":    "#911eb4",
    "High-End": "#e6194b",
    "Mid-Range":"#f58231",
}

# Define stacks of hardware configurations to evaluate
# STACKS = {
#     # "H200-DDR5-6000-NVMeT700R0-NVLink6": {
#     #     "gpu": GPUS["H200"],
#     #     "dram": DRAMS["DDR5-6000"],
#     #     "disk": DISKS["NVMeT700R0"],
#     #     "link": LINKS["NVLink6"],
#     #     "gpu_count": 4,
#     #     "dram_count": 4,
#     # },
#     # "H100-DDR5-6000-NVMeT700R0-NVLink6": {
#     #     "gpu": GPUS["H100"],
#     #     "dram": DRAMS["DDR5-6000"],
#     #     "disk": DISKS["NVMeT700R0"],
#     #     "link": LINKS["NVLink6"],
#     #     "gpu_count": 4,
#     #     "dram_count": 10,
#     # },
#     # "A100-DDR5-6000-NVMeT700R5-NVLink6": {
#     #     "gpu": GPUS["A100"],
#     #     "dram": DRAMS["DDR5-6000"],
#     #     "disk": DISKS["NVMeT700R5"],
#     #     "link": LINKS["NVLink6"],
#     #     "gpu_count": 4,
#     #     "dram_count": 10,
#     # },
#     "H200-DDR4-3200-NVMe980-NVLink3": {
#         "gpu": GPUS["H200"],
#         "dram": DRAMS["DDR4-3200"],
#         "disk": DISKS["NVMe980"],
#         "link": LINKS["NVLink3"],
#         "gpu_count": 1,
#         "dram_count": 2,
#     },
#     "H100-DDR5-6000-NVMe980-NVLink3": {
#         "gpu": GPUS["H100"],
#         "dram": DRAMS["DDR5-6000"],
#         "disk": DISKS["NVMe980"],
#         "link": LINKS["NVLink3"],
#         "gpu_count": 1,
#         "dram_count": 2,
#     },
#     "A100-DDR5-6000-NVMeT700R5-NVLink5": {
#         "gpu": GPUS["A100"],
#         "dram": DRAMS["DDR5-6000"],
#         "disk": DISKS["NVMeT700R5"],
#         "link": LINKS["NVLink5"],
#         "gpu_count": 1,
#         "dram_count": 2,
#     },
# }
STACKS = {
    "High-End (DDR5+NVMe)": {
        "gpu": GPUS["High-End"],
        "dram": DRAMS["DDR5"],
        "disk": DISKS["NVMe"],
        "link": LINKS["NVLink"],
        "gpu_count": 1,
        "dram_count": 2,
    },
    "Mid-Range (DDR5+NVMe)": {
        "gpu": GPUS["Mid-Range"],
        "dram": DRAMS["DDR5"],
        "disk": DISKS["NVMe"],
        "link": LINKS["NVLink"],
        "gpu_count": 1,
        "dram_count": 2,
    },
    "Mid-Range (DDR4+NVMe)": {
        "gpu": GPUS["Mid-Range"],
        "dram": DRAMS["DDR4"],
        "disk": DISKS["NVMe"],
        "link": LINKS["NVLink"],
        "gpu_count": 1,
        "dram_count": 2,
    },
    "Mid-Range (DDR4+SATA)": {
        "gpu": GPUS["Mid-Range"],
        "dram": DRAMS["DDR4"],
        "disk": DISKS["SATA"],
        "link": LINKS["PCIe"],
        "gpu_count": 1,
        "dram_count": 2,
    },
}


# Helper function to convert byte capacity to number of blocks
def cap_to_blocks(byte_cap):
    return byte_cap / CONFIG["BYTES_PER_BLOCK"]


# Helper function to convert number of blocks to byte capacity
def fmt_bytes_from_blocks(n_blocks):
    b = n_blocks * CONFIG["BYTES_PER_BLOCK"]
    if b >= 1e12:
        return f"{b / 1e12:.1f} TB"
    if b >= 1e9:
        return f"{b / 1e9:.0f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.0f} MB"
    return f"{b:.0f} B"


# Helper function to format time in human-readable units
def fmt_time(s):
    if s <= 0:
        return "0"
    if s < 1e-6:
        return f"{s * 1e9:.0f} ns"
    if s < 1e-3:
        return f"{s * 1e6:.0f} µs"
    if s < 1:
        return f"{s * 1e3:.0f} ms"
    if s < 60:
        return f"{s:.2f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    if s < 86400:
        return f"{s / 3600:.1f} hr"
    if s == 86400:
        return "1 day"
    return f"{s / 86400:.1f} days"


# Helper function to get a color for a given label
def get_color(label):
    color_palette = plt.get_cmap("tab10")
    color_index = abs(hash(label)) % color_palette.N
    return color_palette(color_index)


# Function to calculate the storage curve for a given stack configuration
def build_storage_curve(xs_blocks, model_params, stack: dict):
    gpu = stack["gpu"]
    dram = stack["dram"]
    disk = stack["disk"]
    link = stack["link"]

    vram_cap = stack["gpu_count"] * gpu.hbm_capacity - 2 * model_params
    if vram_cap <= 0:
        return np.full_like(xs_blocks, np.inf, dtype=float)
    
    dram_cap = stack["dram_count"] * dram.capacity

    vram_blk = cap_to_blocks(vram_cap)
    dram_blk = cap_to_blocks(dram_cap)
    disk_blk = cap_to_blocks(disk.capacity)

    BYTES = CONFIG["BYTES_PER_BLOCK"]

    times = np.empty_like(xs_blocks, dtype=float)
    for i, n in enumerate(xs_blocks):

        if n <= vram_blk:
            # All data fits in VRAM — only HBM access cost
            t = (n * BYTES) / gpu.hbm_bandwidth

        elif n <= vram_blk + dram_blk:
            # Overflow spills into DRAM: disk → DRAM → link → GPU
            overflow = (n - vram_blk) * BYTES
            t = (
                (vram_blk * BYTES) / gpu.hbm_bandwidth    # VRAM portion
                + overflow / dram.bandwidth               # DRAM read
                + overflow / link.bandwidth               # link transfer to GPU
            )

        elif n <= vram_blk + dram_blk + disk_blk:
            # Overflow spills into disk: disk data stages through DRAM then link
            dram_bytes = dram_blk * BYTES
            overflow = (n - vram_blk - dram_blk) * BYTES
            t = (
                (vram_blk * BYTES) / gpu.hbm_bandwidth    # VRAM portion
                + dram_bytes / dram.bandwidth             # DRAM read
                + dram_bytes / link.bandwidth             # DRAM → GPU link
                + overflow / disk.bandwidth               # disk read
                + overflow / dram.bandwidth               # disk data stages through DRAM
                + overflow / link.bandwidth               # disk data → GPU link
            )

        else:
            # Exceeds all storage tiers — physically impossible
            t = np.inf

        times[i] = t
    return times

# Plot function
def make_plot(
    figsize=(6, 3.5),
    dpi=700,
    output="plot.png",
):
    max_blocks = 500_000
    min_blocks = 500
    model_params = CONFIG["MODEL_PARAMS"]

    # create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    #ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min_blocks, max_blocks)

    # set x space
    # xs = np.logspace(np.log10(min_blocks), np.log10(max_blocks), CONFIG["NPOINTS"]).astype(np.int64)
    xs = np.linspace(min_blocks, max_blocks, CONFIG["NPOINTS"]).astype(np.int64)

    # create legend items list to store the legend entries for each stack
    legend_items = []
    computes_hitmap = {}

    # loop through model parameters and stacks to plot compute time against number of blocks
    for stack_name, stack in STACKS.items():
        gpu = stack["gpu"]
        color = get_color(stack_name)

        # plot compute information for the GPU
        if gpu.name not in computes_hitmap:
            computes_hitmap[gpu.name] = True

            compute_label = f"[Compute] {stack['gpu_count']}x{gpu.name}"
            legend_items.insert(0,
                Line2D([0], [0], color=color, lw=1.6, linestyle=":", label=compute_label)
            )

            gpu_compute_band = gpu.gpu_compute_band(model_params, gpu_count=stack["gpu_count"], eta=CONFIG["GPU_ETA"])
            ys = [x * gpu_compute_band for x in xs]

            ax.plot(xs, ys, color=color, lw=1.6, linestyle=":", label=compute_label)

        # plot the storage line information for the stack
        stack_cost = stack["gpu"].cost + stack["dram"].cost * stack["dram_count"] + stack["disk"].cost + stack["link"].cost
        storage_label = f"[Restore] {stack_name} (${stack_cost:.0f})"
        legend_items.append(
            Line2D([0], [0], color=color, lw=1.6, linestyle="-", label=storage_label)
        )

        ys = build_storage_curve(xs, model_params, stack)
        ax.plot(xs, ys, color=color, lw=1.6, linestyle="-", label=storage_label)

    # ── Axes ──────────────────────────────────────────────────────────────────
    x_ticks = [16, 16_000, 64_000, 128_000, 200_000, 300_000, 400_000, 500_000]
    x_ticks = [v for v in x_ticks if min_blocks <= v <= max_blocks]

    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Number of Tokens", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # Top axis: data volume
    ax2 = ax.twiny()
    # ax2.set_xscale("log")
    ax2.set_xlim(min_blocks, max_blocks)
    ax2.set_xticks(x_ticks)
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: fmt_bytes_from_blocks(v))
    )
    # ax2.set_xlabel("Data volume", fontsize=8, labelpad=6)
    ax2.tick_params(axis="x", labelsize=8)

    # Y-axis
    y_ticks = [
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1,
        10,
        60,
        600,
        3600,
    ]
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_time(v)))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.set_ylabel("Time (Storage restore  vs.  GPU compute)", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # ax.set_title(
    #     "Multi-tier Storage Restore vs GPU Recompute",
    #     fontsize=11,
    #     pad=10,
    # )

    # plot legends
    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=8,
        framealpha=0.94,
        edgecolor="#cccccc",
        handlelength=3.0,
    )

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {output}")


# ── Permutation scatter plot ──────────────────────────────────────────────────

# Keys included in the combinatorial sweep
_PERM_GPU_KEYS  = ["H200", "H100", "A100", "RTX6000", "V100", "A5000"]
_PERM_DRAM_KEYS = ["DDR5-6000", "DDR5-5600", "DDR4-3200", "DDR4-2133", "DDR3-1600"]
_PERM_DISK_KEYS = ["HDD", "X110", "M550", "NVMe980", "NVMeT700", "NVMeT700R0", "NVMeT700R5"]
_PERM_LINK_KEYS = ["PCIe3", "PCIe4", "PCIe5", "NVLink3", "NVLink4", "NVLink5", "NVLink6"]

# One color per GPU — all (GPU, *, *) dots share the GPU's color
_PERM_GPU_COLORS = {
    "H200":    "#e6194b",
    "H100":    "#f58231",
    "A100":    "#bfbf00",
    "RTX6000": "#3cb44b",
    "V100":    "#4363d8",
    "A5000":   "#911eb4",
}


def _pick_link(gpu_key):
    if gpu_key in ("H200", "H100", "A100"):
        return LINKS["NVLink4"]
    return LINKS["PCIe4"]


def _storage_time_at(n_blocks, model_params, gpu, dram, disk, link, dram_count=2):
    BYTES = CONFIG["BYTES_PER_BLOCK"]
    vram_cap = gpu.hbm_capacity - 2 * model_params
    if vram_cap <= 0:
        return np.inf

    vram_blk = cap_to_blocks(vram_cap)
    dram_blk = cap_to_blocks(dram_count * dram.capacity)
    disk_blk = cap_to_blocks(disk.capacity)

    if n_blocks <= vram_blk:
        return (n_blocks * BYTES) / gpu.hbm_bandwidth

    if n_blocks <= vram_blk + dram_blk:
        overflow = (n_blocks - vram_blk) * BYTES
        return (
            (vram_blk * BYTES) / gpu.hbm_bandwidth
            + overflow / dram.bandwidth
            + overflow / link.bandwidth
        )

    if n_blocks <= vram_blk + dram_blk + disk_blk:
        dram_bytes = dram_blk * BYTES
        overflow = (n_blocks - vram_blk - dram_blk) * BYTES
        return (
            (vram_blk * BYTES) / gpu.hbm_bandwidth
            + dram_bytes / dram.bandwidth
            + dram_bytes / link.bandwidth
            + overflow / disk.bandwidth
            + overflow / dram.bandwidth
            + overflow / link.bandwidth
        )

    return np.inf


def make_permutation_plot(figsize=(8, 4.5), dpi=700, output="plot_permutations.png"):
    import itertools

    model_params = CONFIG["MODEL_PARAMS"]

    x_ticks = [16_000, 64_000, 128_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000]
    y_ticks  = [1e-3, 1e-2, 1e-1, 1, 10, 60, 600, 3600]

    _, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_yscale("log")
    ax.set_xlim(x_ticks[0], x_ticks[-1])

    legend_items = []

    # Collect storage times across ALL permutations per tick (one merged pool)
    data = {x: [] for x in x_ticks}
    total_options = 0
    for gpu_key, dram_key, disk_key, link_key in itertools.product(
        _PERM_GPU_KEYS, _PERM_DRAM_KEYS, _PERM_DISK_KEYS, _PERM_LINK_KEYS
    ):
        gpu  = GPUS[gpu_key]
        dram = DRAMS[dram_key]
        disk = DISKS[disk_key]
        link = LINKS[link_key]
        total_options += 1
        for x in x_ticks:
            t = _storage_time_at(x, model_params, gpu, dram, disk, link)
            if np.isfinite(t):
                data[x].append(t)

    box_w = (x_ticks[-1] - x_ticks[0]) / len(x_ticks) * 0.35

    # Draw compute lines first (behind boxes) — one per GPU, added to legend
    xs_line = np.linspace(x_ticks[0], x_ticks[-1], 800)
    for gpu_key in _PERM_GPU_KEYS:
        gpu   = GPUS[gpu_key]
        color = _PERM_GPU_COLORS[gpu_key]
        band  = gpu.gpu_compute_band(model_params, gpu_count=1, eta=CONFIG["GPU_ETA"])
        ax.plot(xs_line, xs_line * band, color=color, lw=1.1, linestyle=":",
                alpha=0.85, zorder=2)
        legend_items.append(
            Line2D([0], [0], color=color, lw=1.1, linestyle=":", label=gpu.name)
        )

    # Single merged violin per tick
    vp = ax.violinplot(
        [data[x] for x in x_ticks],
        positions=x_ticks,
        widths=box_w,
        showmedians=True,
        showextrema=True,
    )
    for body in vp["bodies"]:
        body.set_facecolor("#aaaaaa")
        body.set_edgecolor("#555555")
        body.set_alpha(0.5)
        body.set_linewidth(0.6)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp[part].set_color("#555555")
        vp[part].set_linewidth(0.8)
    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(1.4)

    legend_items.append(
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#aaaaaa",
               markeredgecolor="#555555", markersize=8, alpha=0.8,
               label="Storage restore range")
    )

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Number of Tokens", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    ax2 = ax.twiny()
    ax2.set_xlim(x_ticks[0], x_ticks[-1])
    ax2.set_xticks(x_ticks)
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: fmt_bytes_from_blocks(v))
    )
    ax2.tick_params(axis="x", labelsize=8)

    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_time(v)))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.set_ylabel("Time", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.text(0.01, 0.99, f"{total_options} configurations",
            transform=ax.transAxes, fontsize=7, va="top", ha="left", color="#555555")

    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=7,
        framealpha=0.92,
        edgecolor="#cccccc",
        handletextpad=0.4,
    )

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=dpi, bbox_inches="tight")
        print(f"Saved → {output}  ({total_options} configurations plotted)")


if __name__ == "__main__":
    make_plot()
    make_permutation_plot()
