import hashlib

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
    "GPU_ETA": 0.6           # GPU effectiveness factor
}

# Define stacks of hardware configurations to evaluate
STACKS = {
    # "H200-DDR5-6000-NVMeT700R0-NVLink6": {
    #     "gpu": GPUS["H200"],
    #     "dram": DRAMS["DDR5-6000"],
    #     "disk": DISKS["NVMeT700R0"],
    #     "link": LINKS["NVLink6"],
    #     "gpu_count": 4,
    #     "dram_count": 4,
    # },
    # "H100-DDR5-6000-NVMeT700R0-NVLink6": {
    #     "gpu": GPUS["H100"],
    #     "dram": DRAMS["DDR5-6000"],
    #     "disk": DISKS["NVMeT700R0"],
    #     "link": LINKS["NVLink6"],
    #     "gpu_count": 4,
    #     "dram_count": 10,
    # },
    # "A100-DDR5-6000-NVMeT700R5-NVLink6": {
    #     "gpu": GPUS["A100"],
    #     "dram": DRAMS["DDR5-6000"],
    #     "disk": DISKS["NVMeT700R5"],
    #     "link": LINKS["NVLink6"],
    #     "gpu_count": 4,
    #     "dram_count": 10,
    # },
    "H200-DDR4-3200-NVMe980-NVLink3": {
        "gpu": GPUS["H200"],
        "dram": DRAMS["DDR4-3200"],
        "disk": DISKS["NVMe980"],
        "link": LINKS["NVLink3"],
        "gpu_count": 1,
        "dram_count": 2,
    },
    "H100-DDR5-6000-NVMe980-NVLink3": {
        "gpu": GPUS["H100"],
        "dram": DRAMS["DDR5-6000"],
        "disk": DISKS["NVMe980"],
        "link": LINKS["NVLink3"],
        "gpu_count": 1,
        "dram_count": 2,
    },
    "A100-DDR5-6000-NVMeT700R5-NVLink5": {
        "gpu": GPUS["A100"],
        "dram": DRAMS["DDR5-6000"],
        "disk": DISKS["NVMeT700R5"],
        "link": LINKS["NVLink5"],
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

    # use a hash of the label to get a consistent color index
    hash_digest = hashlib.md5(label.encode()).hexdigest()
    color_index = int(hash_digest, 24) % color_palette.N
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
                (vram_blk * BYTES) / gpu.hbm_bandwidth   # VRAM portion
                + overflow / dram.bandwidth               # DRAM read
                + overflow / link.bandwidth               # link transfer to GPU
            )

        elif n <= vram_blk + dram_blk + disk_blk:
            # Overflow spills into disk: disk data stages through DRAM then link
            dram_bytes = dram_blk * BYTES
            overflow = (n - vram_blk - dram_blk) * BYTES
            t = (
                (vram_blk * BYTES) / gpu.hbm_bandwidth   # VRAM portion
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
            legend_items.append(
                Line2D([0], [0], color=color, lw=1.6, linestyle=":", label=compute_label)
            )

            gpu_compute_band = gpu.gpu_compute_band(model_params, gpu_count=stack["gpu_count"], eta=CONFIG["GPU_ETA"])
            ys = [x * gpu_compute_band for x in xs]

            ax.plot(xs, ys, color=color, lw=1.6, linestyle=":", label=compute_label)

        # plot the storage line information for the stack
        storage_label = f"[Restore] {stack_name}"
        legend_items.append(
            Line2D([0], [0], color=color, lw=1.6, linestyle="-", label=storage_label)
        )

        ys = build_storage_curve(xs, model_params, stack)
        ax.plot(xs, ys, color=color, lw=1.6, linestyle="-", label=storage_label)

    # ── Axes ──────────────────────────────────────────────────────────────────
    # Ticks aligned to round data sizes:
    # 1=50MB, 20=1GB, 200=10GB, 2000=100GB, 20000=1TB, 200000=10TB,
    # 2000000=100TB, 20000000=1000TB
    x_ticks = [1, 1_000, 100_000, 250_000, 400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000, 1_250_000, 2_000_000, 5_000_000]
    x_ticks = [v for v in x_ticks if min_blocks <= v <= max_blocks]

    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel(f"Number of blocks  (1 block = {CONFIG['TOKENS_PER_BLOCK']} tokens = {CONFIG['BYTES_PER_BLOCK'] / (1024**2):.1f} MB, model parameters = {CONFIG['MODEL_PARAMS'] / (1024**3):.1f}B)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # Top axis: data volume
    ax2 = ax.twiny()
    # ax2.set_xscale("log")
    ax2.set_xlim(min_blocks, max_blocks)
    ax2.set_xticks(x_ticks)
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: fmt_bytes_from_blocks(v))
    )
    ax2.set_xlabel("Data volume", fontsize=8, labelpad=6)
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
    ax.set_ylabel("Time (Storage restore  or  GPU recompute)", fontsize=8)
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


if __name__ == "__main__":
    make_plot()
