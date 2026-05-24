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
    "BYTES_PER_BLOCK": 2e6,  # 2 MB per block
    "MODEL_PARAMS": 8e9,     # 8 billion parameters
    "GPU_ETA": 0.5           # GPU effectiveness factor
}

# GPU family color map — same GPU key = same color
GPU_COLORS = {
    "H200":     "#e6194b",
    "H100":     "#f58231",
    "A100":     "#c0a700",
    "RTX6000":  "#3cb44b",
    "V100":     "#4363d8",
    "A5000":    "#911eb4",
}

# Define stacks of hardware configurations to evaluate
STACKS = {
    "H200-DDR5+NVMe": {
        "gpu": GPUS["H200"],
        "dram": DRAMS["DDR5"],
        "disk": DISKS["NVMe"],
        "link": LINKS["NVLink"],
        "gpu_count": 1,
        "dram_count": 4,
        "color": "#888888",
    },
    "A100-DDR5+NVMe": {
        "gpu": GPUS["A100"],
        "dram": DRAMS["DDR5"],
        "disk": DISKS["NVMe"],
        "link": LINKS["NVLink"],
        "gpu_count": 1,
        "dram_count": 4,
        "color": "#d95f02",
    },
    "A100-DDR4+NVMe": {
        "gpu": GPUS["A100"],
        "dram": DRAMS["DDR4"],
        "disk": DISKS["NVMe"],
        "link": LINKS["NVLink"],
        "gpu_count": 1,
        "dram_count": 4,
        "color": "#1b9e77",
    },
    "A100-DDR4+SATA": {
        "gpu": GPUS["A100"],
        "dram": DRAMS["DDR4"],
        "disk": DISKS["SATA"],
        "link": LINKS["PCIe"],
        "gpu_count": 1,
        "dram_count": 4,
        "color": "#2e2a68",
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


# Function to calculate the GPU compute curve for a given GPU and model parameters
def build_compute_curve(xs_blocks, model_params, gpu, link, gpu_count=1):
    # calculate the effective VRAM capacity after reserving space for model parameters
    model_size_bytes = model_params * 2  # bf16: 2 bytes per param
    vram_cap = gpu.hbm_capacity * gpu_count - model_size_bytes
    if vram_cap <= 0:
        return np.full_like(xs_blocks, np.inf, dtype=float)

    # convert the effective VRAM capacity from bytes to number of blocks
    vram_blk = cap_to_blocks(vram_cap)

    # compute time per token for the GPU
    t_per_token = gpu.gpu_compute_band(model_params, gpu_count=gpu_count, eta=CONFIG["GPU_ETA"])

    # compute time for processing n blocks, considering both memory access and compute time, depending on how many blocks fit in VRAM
    def block_time(n, bandwidth):
        t_mem = (n * CONFIG["BYTES_PER_BLOCK"]) / bandwidth
        t_cmp = n * CONFIG["TOKENS_PER_BLOCK"] * t_per_token
        return max(t_mem, t_cmp)

    # compute the total time for each block count in xs_blocks
    times = np.empty_like(xs_blocks, dtype=float)
    for i, n in enumerate(xs_blocks):
        if n <= vram_blk:
            times[i] = block_time(n, gpu.hbm_bandwidth)
        else:
            t_vram     = block_time(vram_blk, gpu.hbm_bandwidth)
            t_overflow = block_time(n - vram_blk, link.bandwidth)  # PCIe
            times[i]   = t_vram + t_overflow

    return times


# Function to calculate the storage curve for a given stack configuration
def build_storage_curve(xs_blocks, model_params, stack: dict):
    gpu   = stack["gpu"]
    dram  = stack["dram"]
    disk  = stack["disk"]
    link  = stack["link"]

    # calculate the effective VRAM capacity after reserving space for model parameters
    model_size_bytes = model_params * 2  # bf16
    vram_cap = stack["gpu_count"] * gpu.hbm_capacity - model_size_bytes
    if vram_cap <= 0:
        return np.full_like(xs_blocks, np.inf, dtype=float)

    # calculate the effective DRAM capacity based on the number of DRAM modules in the stack
    dram_cap = stack["dram_count"] * dram.capacity
    vram_blk = cap_to_blocks(vram_cap)
    dram_blk = cap_to_blocks(dram_cap)
    BYTES    = CONFIG["BYTES_PER_BLOCK"]

    # compute time per token for the GPU
    t_per_token = gpu.gpu_compute_band(
        model_params, gpu_count=stack["gpu_count"], eta=CONFIG["GPU_ETA"]
    )

    # compute the total time for each block count in xs_blocks, considering the tiered storage hierarchy and the compute time,
    # using a roofline model to overlap compute and transfer where possible
    times = np.empty_like(xs_blocks, dtype=float)
    for i, n in enumerate(xs_blocks):
        # compute time for n blocks, assuming all blocks are in VRAM (best case)
        t_compute = n * CONFIG["TOKENS_PER_BLOCK"] * t_per_token

        # memory transfer time for n blocks, depending on how many blocks fit in each tier of the storage hierarchy (VRAM → DRAM → Disk)
        if n <= vram_blk:
            t_mem = (n * BYTES) / gpu.hbm_bandwidth

        elif n <= vram_blk + dram_blk:
            dram_overflow = (n - vram_blk) * BYTES
            t_mem = (
                (vram_blk * BYTES) / gpu.hbm_bandwidth
                + dram_overflow / min(dram.bandwidth, link.bandwidth)
            )

        else:
            dram_full     = dram_blk * BYTES
            disk_overflow = (n - vram_blk - dram_blk) * BYTES
            t_mem = (
                (vram_blk * BYTES) / gpu.hbm_bandwidth
                + dram_full     / min(dram.bandwidth, link.bandwidth)
                + disk_overflow / min(disk.bandwidth, link.bandwidth)
            )

        times[i] = t_compute + t_mem  # roofline: overlap compute and transfer

    return times


# Plot function
def make_plot(
    figsize=(6, 3.5),
    dpi=700,
    output="tier_caps.pdf",
):
    max_blocks = 500_000
    min_blocks = 500
    model_params = CONFIG["MODEL_PARAMS"]

    # create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_yscale("log")
    ax.set_xlim(min_blocks, max_blocks)

    # set x space
    xs = np.linspace(min_blocks, max_blocks, CONFIG["NPOINTS"]).astype(np.int64)

    # create legend items list to store the legend entries for each stack
    legend_items = []
    computes_hitmap = {}

    # loop through model parameters and stacks to plot compute time against number of blocks
    for stack_name, stack in STACKS.items():
        gpu = stack["gpu"]

        # plot compute information for the GPU
        if gpu.name not in computes_hitmap:
            computes_hitmap[gpu.name] = True

            compute_label = f"{stack['gpu_count']}x{gpu.name} (${gpu.cost:.0f})"
            legend_items.insert(0,
                Line2D([0], [0], color=GPU_COLORS[gpu.name], lw=1.6, linestyle=":", label=compute_label)
            )

            ys = build_compute_curve(xs, model_params, gpu, stack["link"], gpu_count=stack["gpu_count"])
            ax.plot(xs, ys, color=GPU_COLORS[gpu.name], lw=1.6, linestyle=":", label=compute_label)

        # plot the storage line information for the stack
        stack_cost = stack["gpu"].cost + stack["dram"].cost * stack["dram_count"] + stack["disk"].cost + stack["link"].cost
        storage_label = f"{stack_name} (${stack_cost:.0f})"
        legend_items.append(
            Line2D([0], [0], color=stack["color"], lw=1.6, linestyle="-", label=storage_label)
        )

        ys = build_storage_curve(xs, model_params, stack)
        ax.plot(xs, ys, color=stack["color"], lw=1.2, linestyle="-", label=storage_label)

    # ── Axes ──────────────────────────────────────────────────────────────────
    x_ticks = [16, 16_000, 64_000, 128_000, 200_000, 300_000, 400_000, 500_000]
    x_ticks = [v for v in x_ticks if min_blocks <= v <= max_blocks]

    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("Context Length", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

    # Top axis: data volume
    ax2 = ax.twiny()
    ax2.set_xlim(min_blocks, max_blocks)
    ax2.set_xticks(x_ticks)
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: fmt_bytes_from_blocks(v))
    )
    ax2.tick_params(axis="x", labelsize=8)

    # Y-axis
    y_ticks = [
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

    # plot some circles with labels in plot to reference in presentations
    points_of_interest = [
        {'x': 16_000, 'y': 5, 'label': 'A', 'color': 'blue'},
        {'x': 64_000, 'y': 35, 'label': 'B', 'color': 'green'},
        {'x': 128_000, 'y': 65, 'label': 'C', 'color': 'orange'},
        {'x': 400_000, 'y': 105, 'label': 'D', 'color': 'red'},
    ]
    for point in points_of_interest:
        ax.scatter(
            point['x'],
            point['y'],
            facecolors='white',
            edgecolors=point['color'],
            s=220,
            linewidths=1.2,
            zorder=5,
        )
        ax.text(
            point['x'],
            point['y'],
            point['label'],
            color=point['color'],
            fontsize=9,
            fontweight='bold',
            ha='center',
            va='center',
            zorder=6,
        )

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
        plt.savefig(output, bbox_inches="tight")
        print(f"Saved → {output}")


# Keys included in the combinatorial sweep
_PERM_GPU_KEYS  = ["H200", "H100", "A100", "RTX6000", "V100", "A5000"]
_PERM_DRAM_KEYS = ["DDR5-6000", "DDR5-7200", "DDR5-5600", "DDR4-3200", "DDR4-2133", "DDR3-1600"]
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


def make_permutation_plot(figsize=(8, 4.5), dpi=700, output="tiers_configuration.pdf"):
    import itertools

    model_params = CONFIG["MODEL_PARAMS"]

    x_ticks = [16_000, 64_000, 128_000, 200_000, 250_000, 300_000, 350_000, 400_000, 450_000, 500_000]
    y_ticks  = [1, 10, 60, 600, 3600, 3600*4]

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
            # Build a temporary stack dict to reuse `build_storage_curve` logic
            stack = {
                "gpu": gpu,
                "dram": dram,
                "disk": disk,
                "link": link,
                "gpu_count": 1,
                "dram_count": 2,
            }
            # build_storage_curve returns compute+restore time for given x
            t_arr = build_storage_curve(np.array([x], dtype=np.int64), model_params, stack)
            t = float(t_arr[0])
            if np.isfinite(t):
                data[x].append(t)

    box_w = (x_ticks[-1] - x_ticks[0]) / len(x_ticks) * 0.35

    # Draw compute lines first (behind boxes) — one per GPU, added to legend
    xs_line = np.linspace(x_ticks[0], x_ticks[-1], 800).astype(np.int64)
    for gpu_key in _PERM_GPU_KEYS:
        gpu = GPUS[gpu_key]
        color = _PERM_GPU_COLORS[gpu_key]
        # Use the same compute-curve builder so permutation compute lines match main plot.
        # Provide a high-bandwidth link so the line reflects compute-dominated behavior.
        high_bw_link = LINKS.get("NVLink6", LINKS.get("NVLink", list(LINKS.values())[0]))
        ys_compute = build_compute_curve(xs_line, model_params, gpu, high_bw_link, gpu_count=1)
        ax.plot(xs_line, ys_compute, color=color, lw=1.1, linestyle=":",
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
    ax.set_xlabel("Context Length", fontsize=9)
    ax.tick_params(axis="x", labelsize=9)

    ax2 = ax.twiny()
    ax2.set_xlim(x_ticks[0], x_ticks[-1])
    ax2.set_xticks(x_ticks)
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: fmt_bytes_from_blocks(v))
    )
    ax2.tick_params(axis="x", labelsize=9)

    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_time(v)))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, which="major", linestyle="--", linewidth=0.45, alpha=0.35)
    ax.set_ylabel("Time (Storage restore  vs.  GPU compute)", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)

    ax.text(0.01, 0.99, f"{total_options} configurations",
            transform=ax.transAxes, fontsize=8, va="top", ha="left", color="#555555")

    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=8,
        framealpha=0.92,
        edgecolor="#cccccc",
        handletextpad=0.4,
    )

    plt.tight_layout()

    if output:
        plt.savefig(output, bbox_inches="tight")
        print(f"Saved → {output}  ({total_options} configurations plotted)")


if __name__ == "__main__":
    make_plot()
    make_permutation_plot()
