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

# global configuration for the simulator
CONFIG = {
    "NPOINTS": 5000,         # Number of points to plot
    "TOKENS_PER_BLOCK": 16,  # Number of tokens per block
    "BYTES_PER_BLOCK": 2e6,  # 2 MB per block
    "MODEL_PARAMS": 8e9,     # 8 billion parameters
    "MODEL_LAYERS": 32,      # transformer blocks (Llama-3 8B shape)
    "MODEL_DIM": 4096,       # hidden size
    "GPU_ETA": 0.5,          # achieved fraction of dense peak during prefill
    "RESTORE_HBM_PASSES": 2, # HBM passes over restored KV: land it, then read it
    "KV_COMPRESSION": 0.1,   # KV stored compressed (CacheGen reports 3.5-4.3x)
    "DECODE_RATE": 100e9,    # GPU decode of compressed KV, bytes/s of expanded cache
    "DECODE_RATE_HBM": 2.039e12,  # ... on A100-class HBM (~5% of it); scaled by bandwidth
}

# GPU family color map (same GPU key = same color)
GPU_COLORS = {
    "H200": "#e6194b",
    "H100": "#f58231",
    "A100": "#c0a700",
    "RTX6000": "#3cb44b",
    "V100": "#4363d8",
    "A5000": "#911eb4",
}

# define stacks of hardware configurations to evaluate
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
    if b == 0:
        return "0"
    
    b /= 1e9  # convert to GB

    # use standard scientific notation with one digit before the decimal
    exp = int(np.floor(np.log10(abs(b))))
    mantissa = b / (10**exp)

    # format mantissa with up to 15 significant digits, then trim trailing zeros
    mantissa_str = f"{mantissa:.15f}".rstrip("0").rstrip(".")
    if exp == 0:
        return mantissa_str
    
    return f"{mantissa_str}e{exp}"


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


# Helper function to format tick values in engineering notation
def fmt_engineering(v):
    if v == 0:
        return "0"
    
    # use standard scientific notation with one digit before the decimal
    exp = int(np.floor(np.log10(abs(v))))
    mantissa = v / (10**exp)

    # format mantissa with up to 15 significant digits, then trim trailing zeros
    mantissa_str = f"{mantissa:.15f}".rstrip("0").rstrip(".")
    if exp == 0:
        return mantissa_str
    
    return f"{mantissa_str}e{exp}"


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
    t_per_token = gpu.gpu_compute_band(
        model_params, gpu_count=gpu_count, eta=CONFIG["GPU_ETA"]
    )

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
            t_vram = block_time(vram_blk, gpu.hbm_bandwidth)
            t_overflow = block_time(n - vram_blk, link.bandwidth)  # PCIe
            times[i] = t_vram + t_overflow

    return times


# Function to calculate the storage restore curve
def build_storage_curve(xs_blocks, model_params, stack: dict):
    gpu = stack["gpu"]
    dram = stack["dram"]
    disk = stack["disk"]
    link = stack["link"]

    # calculate the effective VRAM capacity after reserving space for model parameters
    model_size_bytes = model_params * 2  # bf16
    vram_cap = stack["gpu_count"] * gpu.hbm_capacity - model_size_bytes
    if vram_cap <= 0:
        return np.full_like(xs_blocks, np.inf, dtype=float)

    # calculate the effective DRAM capacity based on the number of DRAM modules in the stack
    dram_cap = stack["dram_count"] * dram.capacity
    vram_blk = cap_to_blocks(vram_cap)
    dram_blk = cap_to_blocks(dram_cap)
    BYTES = CONFIG["BYTES_PER_BLOCK"]

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

        # memory transfer time for n blocks, depending on how many blocks fit in each tier of the storage hierarchy (VRAM to DRAM to Disk)
        if n <= vram_blk:
            t_mem = (n * BYTES) / gpu.hbm_bandwidth

        elif n <= vram_blk + dram_blk:
            dram_overflow = (n - vram_blk) * BYTES
            t_mem = (vram_blk * BYTES) / gpu.hbm_bandwidth + dram_overflow / min(
                dram.bandwidth, link.bandwidth
            )

        else:
            dram_full = dram_blk * BYTES
            disk_overflow = (n - vram_blk - dram_blk) * BYTES
            t_mem = (
                (vram_blk * BYTES) / gpu.hbm_bandwidth
                + dram_full / min(dram.bandwidth, link.bandwidth)
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
            legend_items.insert(
                0,
                Line2D(
                    [0],
                    [0],
                    color=GPU_COLORS[gpu.name],
                    lw=1.6,
                    linestyle=":",
                    label=compute_label,
                ),
            )

            ys = build_compute_curve(
                xs, model_params, gpu, stack["link"], gpu_count=stack["gpu_count"]
            )
            ax.plot(
                xs,
                ys,
                color=GPU_COLORS[gpu.name],
                lw=1.6,
                linestyle=":",
                label=compute_label,
            )

        # plot the storage line information for the stack
        stack_cost = (
            stack["gpu"].cost
            + stack["dram"].cost * stack["dram_count"]
            + stack["disk"].cost
            + stack["link"].cost
        )
        storage_label = f"{stack_name} (${stack_cost:.0f})"
        legend_items.append(
            Line2D(
                [0],
                [0],
                color=stack["color"],
                lw=1.6,
                linestyle="-",
                label=storage_label,
            )
        )

        ys = build_storage_curve(xs, model_params, stack)
        ax.plot(
            xs, ys, color=stack["color"], lw=1.2, linestyle="-", label=storage_label
        )

    # Axes
    x_ticks = [16, 16_000, 64_000, 128_000, 200_000, 300_000, 400_000, 500_000]
    x_ticks = [v for v in x_ticks if min_blocks <= v <= max_blocks]

    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_engineering(v)))
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
    ax.grid(True, which="major", linestyle="dashed", linewidth=0.45, alpha=0.35)
    ax.set_ylabel("Time\n(Restore vs. Compute)", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # plot some circles with labels in plot to reference in presentations
    points_of_interest = [
        {"x": 16_000, "y": 5, "label": "A", "color": "blue"},
        {"x": 64_000, "y": 35, "label": "B", "color": "green"},
        {"x": 128_000, "y": 65, "label": "C", "color": "orange"},
        {"x": 400_000, "y": 105, "label": "D", "color": "red"},
    ]
    for point in points_of_interest:
        ax.axvline(
            point["x"],
            color=point["color"],
            linewidth=0.8,
            alpha=0.22,
            zorder=1,
        )
        ax.scatter(
            point["x"],
            point["y"],
            facecolors="white",
            edgecolors=point["color"],
            s=220,
            linewidths=1.2,
            zorder=5,
        )
        ax.text(
            point["x"],
            point["y"],
            point["label"],
            color=point["color"],
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
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
_PERM_GPU_KEYS = ["H200", "H100", "A100", "RTX6000", "V100", "A5000"]
_PERM_DRAM_KEYS = [
    "DDR5-6000",
    "DDR5-7200",
    "DDR5-5600",
    "DDR4-3200",
    "DDR4-2133",
    "DDR3-1600",
]
_PERM_DISK_KEYS = [
    "HDD",
    "X110",
    "M550",
    "NVMe980",
    "NVMeT700",
    "NVMeT700R0",
    "NVMeT700R5",
]
_PERM_LINK_KEYS = [
    "PCIe3",
    "PCIe4",
    "PCIe5",
    "NVLink3",
    "NVLink4",
    "NVLink5",
    "NVLink6",
]

# One color per GPU - all (GPU, *, *) dots share the GPU's color
_PERM_GPU_COLORS = {
    "H200": "#e6194b",
    "H100": "#f58231",
    "A100": "#bfbf00",
    "RTX6000": "#3cb44b",
    "V100": "#4363d8",
    "A5000": "#911eb4",
}


def make_permutation_plot(figsize=(8, 3.5), dpi=700, output="tiers_configuration.pdf"):
    import itertools

    model_params = CONFIG["MODEL_PARAMS"]

    x_ticks = [
        16_000,
        64_000,
        128_000,
        200_000,
        250_000,
        300_000,
        350_000,
    ]
    y_ticks = [2, 10, 60, 600, 3600, 3600 * 3]

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
        gpu = GPUS[gpu_key]
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
            t_arr = build_storage_curve(
                np.array([x], dtype=np.int64), model_params, stack
            )
            t = float(t_arr[0])
            if np.isfinite(t):
                data[x].append(t)

    box_w = (x_ticks[-1] - x_ticks[0]) / len(x_ticks) * 0.35

    # Draw compute lines first (behind boxes) - one per GPU, added to legend
    xs_line = np.linspace(x_ticks[0], x_ticks[-1], 800).astype(np.int64)
    for gpu_key in _PERM_GPU_KEYS:
        gpu = GPUS[gpu_key]
        color = _PERM_GPU_COLORS[gpu_key]
        # Use the same compute-curve builder so permutation compute lines match main plot.
        # Provide a high-bandwidth link so the line reflects compute-dominated behavior.
        high_bw_link = LINKS.get(
            "NVLink5", LINKS.get("NVLink", list(LINKS.values())[0])
        )
        ys_compute = build_compute_curve(
            xs_line, model_params, gpu, high_bw_link, gpu_count=1
        )
        ax.plot(
            xs_line,
            ys_compute,
            color=color,
            lw=1.5,
            linestyle=":",
            alpha=0.85,
            zorder=2,
        )
        legend_items.append(
            Line2D([0], [0], color=color, lw=1.3, linestyle=":", label=gpu.name)
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
    vp["cmedians"].set_color("red")
    vp["cmedians"].set_linewidth(4)

    # legend_items.append(
    #     Line2D(
    #         [0],
    #         [0],
    #         marker="s",
    #         color="w",
    #         markerfacecolor="#aaaaaa",
    #         markeredgecolor="#555555",
    #         markersize=8,
    #         alpha=0.8,
    #         label="Storage restore range",
    #     )
    # )

    # Axes
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_engineering(v)))
    ax.set_xlabel("Context Length", fontsize=12)
    ax.tick_params(axis="x", labelsize=11)

    ax2 = ax.twiny()
    ax2.set_xlim(x_ticks[0], x_ticks[-1])
    ax2.set_xticks(x_ticks)
    ax2.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: fmt_bytes_from_blocks(v))
    )
    ax2.tick_params(axis="x", labelsize=11)
    ax2.set_xlabel("Data Volume (GB)", fontsize=12)

    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_time(v)))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(True, which="major", linestyle="dashed", linewidth=0.45, alpha=0.35)
    ax.set_ylabel("Time\n(Restore vs. Compute)", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)

    ax.text(
        0.01,
        0.99,
        f"{total_options} configurations",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        ha="left",
        color="#555555",
    )

    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=12,
        framealpha=0.92,
        edgecolor="#cccccc",
        handletextpad=0.4,
        ncol=3,
    )

    plt.tight_layout()

    if output:
        plt.savefig(output, bbox_inches="tight")
        print(f"Saved → {output}  ({total_options} configurations plotted)")



def _log_violin(ax, datasets, positions, width, face="#aaaaaa", edge="#555555"):
    """Violins of log10(time) on a linear axis.

    ``violinplot`` fits its KDE in data space, so on a log axis a distribution
    spanning decades collapses into a bottom-heavy blob. Fitting on log10(t) and
    labelling the axis with real times keeps the shape readable - callers must
    plot every other series as log10 too.
    """
    vp = ax.violinplot(
        [np.log10(np.asarray(values, dtype=float)) for values in datasets],
        positions=positions,
        widths=width,
        showmedians=True,
        showextrema=True,
    )
    for body in vp["bodies"]:
        body.set_facecolor(face)
        body.set_edgecolor(edge)
        body.set_alpha(0.5)
        body.set_linewidth(0.6)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        vp[part].set_color(edge)
        vp[part].set_linewidth(0.8)
    vp["cmedians"].set_color("red")
    vp["cmedians"].set_linewidth(4)
    return vp


def _log_time_axis(ax, y_ticks, y_lo, y_hi):
    """Label a log10-valued axis with human-readable times."""
    ax.set_ylim(np.log10(y_lo), np.log10(y_hi))
    ax.set_yticks([np.log10(v) for v in y_ticks])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: fmt_time(10**v)))
    ax.yaxis.set_minor_locator(ticker.NullLocator())


# No-cache compute vs. tiered restore
#
# The two curves this figure compares scale differently in the context length N:
#
#   * Prefilling with no KV cache is O(N^2) - every token attends to every
#     earlier token, so the attention term grows quadratically and eventually
#     dominates the linear weight term.
#   * Restoring a cached context is O(N) - it is bytes over the tier path, and
#     the bytes grow linearly with N.
#
# So the lines must cross: a GPU that beats restore on short contexts falls
# behind it on long ones, and a slow GPU is behind for the whole range.


# Function to calculate prefill time with no KV cache (O(N^2) in context length)
def build_prefill_curve(xs_tokens, model_params, gpu, gpu_count=1):
    layers = CONFIG["MODEL_LAYERS"]
    dim = CONFIG["MODEL_DIM"]

    n = np.asarray(xs_tokens, dtype=float)

    # Weights: 2 FLOPs per parameter per token - linear in N.
    flops_weights = 2.0 * model_params * n
    # Causal attention (QK^T and AV over the whole prefix) - quadratic in N.
    flops_attention = 2.0 * layers * dim * n**2

    throughput = gpu.peak_flops * gpu_count * CONFIG["GPU_ETA"]
    return (flops_weights + flops_attention) / throughput


# Function to calculate restore time from the storage tiers (O(N) in context length)
def build_restore_curve(xs_tokens, dram, disk, link, gpu, gpu_count=1):
    """Seconds to bring a cached context back over disk -> DRAM -> link -> GPU.

    Two things real systems do decide the shape of this model:

    * The cache is stored **compressed** (CacheGen reports 3.5-4.3x), so the tier
      path moves ``KV_COMPRESSION`` times fewer bytes - which is what makes the
      cheap tiers usable at all - but the GPU has to decode the stream back into
      a usable cache. That decode is the one restore stage that scales with the
      GPU rather than with the storage stack.
    * Serving stacks **pipeline** the two: LMCache gives inference and data
      movement separate CUDA streams and transforms layer i+1 into pages while
      layer i computes; CacheGen overlaps decoding chunk i-1 with transmitting
      chunk i. So the stages overlap, and a restore costs ``max`` of the two
      paths, not their sum.

    Everything here is linear in N, so a restore stays O(N); which side binds is
    what changes - a fast stack under a weak GPU is decode-bound, a slow disk
    under any GPU is transfer-bound.
    """
    n = np.asarray(xs_tokens, dtype=float)
    kv_bytes = (n / CONFIG["TOKENS_PER_BLOCK"]) * CONFIG["BYTES_PER_BLOCK"]
    wire_bytes = kv_bytes / CONFIG["KV_COMPRESSION"]

    # Storage path: disk read, staged through DRAM, shipped over the link. The
    # link is capped by the GPU's own interconnect - a V100 cannot be fed over
    # NVLink 6, and an Ada card has no NVLink at all - so the same catalog link
    # delivers different bandwidth depending on the part it is bolted to.
    link_bandwidth = gpu.effective_link_bandwidth(link.bandwidth) * gpu_count
    seconds_per_byte = 1.0 / disk.bandwidth + 1.0 / dram.bandwidth + 1.0 / link_bandwidth
    t_path = wire_bytes * seconds_per_byte

    # GPU side. Decode the compressed stream back into a full KV cache. The
    # kernel is elementwise and memory-bound, so the rate tracks HBM bandwidth,
    # not peak FLOPS - the catalog's FLOPS mix number formats (sparse FP8 for
    # H200 against FP32 for the workstation parts) and would spread the GPUs by
    # 73x on a term that is really just memory traffic.
    decode_rate = (
        CONFIG["DECODE_RATE"]
        * gpu.hbm_bandwidth
        * gpu_count
        / CONFIG["DECODE_RATE_HBM"]
    )
    t_decode = kv_bytes / decode_rate
    # HBM traffic for the restored blocks: land them, then read them back.
    t_hbm = CONFIG["RESTORE_HBM_PASSES"] * kv_bytes / (gpu.hbm_bandwidth * gpu_count)
    # Attention for one token against the restored prefix (QK^T + AV).
    flops_attention = 4.0 * CONFIG["MODEL_LAYERS"] * CONFIG["MODEL_DIM"] * n
    t_attention = flops_attention / (gpu.peak_flops * gpu_count * CONFIG["GPU_ETA"])

    return np.maximum(t_path, t_decode + t_hbm + t_attention)


# Storage stack swept for the restore distribution. The GPU dropped out of the
# restore math, so DRAM and disk carry the tier count: 12 x 21 x 7 = 1764.
_RESTORE_DRAM_KEYS = [
    "DDR3-1600",     # 20 GB/s
    "DDR4-2133",     # 25 GB/s
    "DDR4-2400",     # 26 GB/s
    "DDR4-2666",     # 28 GB/s
    "DDR4-3200",     # 30 GB/s
    "DDR5-4800",     # 55 GB/s
    "DDR5-5600",     # 62 GB/s
    "DDR5-6400",     # 75 GB/s
    "DDR5-7200",     # 81 GB/s
    "DDR5-6000",     # 85 GB/s
    "DDR5-8000",     # 95 GB/s
    "LPDDR5X-8533",  # 102 GB/s
]
_RESTORE_DISK_KEYS = [
    "HDD5400",       # 0.08 GB/s
    "HDD",           # 0.1
    "HDDR0x2",       # 0.2
    "X110",          # 0.3
    "M550",          # 0.4
    "HDDR0x5",       # 0.5
    "SATA870",       # 0.56
    "SATAR0x4",      # 1.8
    "NVMe980",       # 2
    "NVMeGen3R0",    # 4
    "NVMeT700",      # 5
    "NVMe990",       # 7
    "NVMeT700R0",    # 10
    "NVMeGen5",      # 12
    "NVMeT700R5",    # 20
    "NVMeGen4R0x4",  # 24
    "RDMADram",      # 25
    "NVMeGen5R0x4",  # 45
    "CXLFlash",      # 64
    "NVMeGen5R0x8",  # 80
    "CXLMem",        # 128
]
_RESTORE_LINK_KEYS = _PERM_LINK_KEYS

# Context lengths on the x axis, in tokens (each doubling is one tick).
_RESTORE_X_TOKENS = [16e3, 32e3, 64e3, 128e3, 256e3, 512e3, 1e6, 2e6]
_RESTORE_Y_TICKS = [0.1, 1, 10, 60, 600, 3600, 3600 * 4]
_RESTORE_Y_MAX = 3600 * 4  # top of the axis: 4 hours


# Helper function to format the KV volume a context occupies
def fmt_gb_from_tokens(n_tokens):
    blocks = n_tokens / CONFIG["TOKENS_PER_BLOCK"]
    return f"{blocks * CONFIG['BYTES_PER_BLOCK'] / 1e9:g}"


def make_no_cache_vs_restore_plots(
    figsize=(8, 3.5), dpi=700, output_prefix="tiers_configuration"
):
    """One figure per GPU: that GPU's no-cache prefill against every stack.

    The violins are identical in all six figures - the restore distribution has
    no GPU in it - so flipping between the plots reads as swapping the GPU under
    a fixed set of 1764 storage deployments. Both axes are plotted as log10 so
    the violin KDE keeps its shape on a log scale; ticks carry the real values.
    """
    import itertools

    model_params = CONFIG["MODEL_PARAMS"]
    x_tokens = np.array(_RESTORE_X_TOKENS, dtype=float)
    x_pos = np.log10(x_tokens)

    # Restore distribution: one sample per (DRAM, disk, link) deployment
    # The GPU takes part in a restore, so each GPU gets its own distribution.
    stacks = list(
        itertools.product(_RESTORE_DRAM_KEYS, _RESTORE_DISK_KEYS, _RESTORE_LINK_KEYS)
    )
    deployments = len(stacks)
    per_gpu_data = {}
    for gpu_key in _PERM_GPU_KEYS:
        data = [[] for _ in x_tokens]
        for dram_key, disk_key, link_key in stacks:
            times = build_restore_curve(
                x_tokens,
                DRAMS[dram_key],
                DISKS[disk_key],
                LINKS[link_key],
                GPUS[gpu_key],
                gpu_count=1,
            )
            for index, t in enumerate(times):
                if np.isfinite(t) and t > 0:
                    data[index].append(float(t))
        per_gpu_data[gpu_key] = data

    # Compute lines: smooth, so they show the quadratic bend
    xs_line = np.logspace(np.log10(x_tokens[0]), np.log10(x_tokens[-1]), 400)
    compute_curves = {
        gpu_key: build_prefill_curve(xs_line, model_params, GPUS[gpu_key], gpu_count=1)
        for gpu_key in _PERM_GPU_KEYS
    }

    # Limits come from the restore spread alone - it is the same in all six
    # figures, so they stay comparable, and a slow GPU's line simply runs off
    # the top instead of stretching every panel to fit it.
    all_values = [
        v for data in per_gpu_data.values() for values in data for v in values
    ]
    y_lo, y_hi = min(all_values) * 0.6, float(_RESTORE_Y_MAX)

    box_w = float(np.diff(x_pos).min()) * 0.42
    outputs = []

    for gpu_key in _PERM_GPU_KEYS:
        gpu = GPUS[gpu_key]
        color = _PERM_GPU_COLORS[gpu_key]

        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set_xlim(x_pos[0] - box_w, x_pos[-1] + box_w)

        # Restore spread: original grey bodies, red medians.
        _log_violin(ax, per_gpu_data[gpu_key], x_pos, box_w)

        # No-cache prefill for this GPU.
        ax.plot(
            np.log10(xs_line),
            np.log10(compute_curves[gpu_key]),
            color=color,
            lw=2.0,
            linestyle=":",
            zorder=4,
        )

        legend_items = [
            Line2D([0], [0], color=color, lw=1.6, linestyle=":", label=gpu.name)
        ]

        # Axes
        ax.set_xticks(x_pos)
        ax.set_xticklabels([fmt_engineering(v) for v in x_tokens])
        ax.set_xlabel("Context Length (Log)", fontsize=12)
        ax.tick_params(axis="x", labelsize=11)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([fmt_gb_from_tokens(v) for v in x_tokens])
        ax2.tick_params(axis="x", labelsize=11)
        ax2.set_xlabel("Data Volume (GB)", fontsize=12)

        _log_time_axis(ax, _RESTORE_Y_TICKS, y_lo, y_hi)
        ax.grid(True, which="major", linestyle="dashed", linewidth=0.45, alpha=0.35)
        ax.set_ylabel("Time (Log)", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)

        ax.text(
            0.01,
            0.99,
            f"{deployments} configurations",
            transform=ax.transAxes,
            fontsize=11,
            va="top",
            ha="left",
            color="#555555",
        )

        ax.legend(
            handles=legend_items,
            loc="lower right",
            fontsize=12,
            framealpha=0.92,
            edgecolor="#cccccc",
            handletextpad=0.4,
            handlelength=3.0,
        )

        plt.tight_layout()

        output = f"{output_prefix}_{gpu_key.lower()}.pdf"
        plt.savefig(output, bbox_inches="tight")
        plt.close()
        outputs.append(output)
        print(f"Saved \u2192 {output}  ({deployments} storage deployments)")

    return outputs


if __name__ == "__main__":
    # make_plot()
    # make_permutation_plot()
    make_no_cache_vs_restore_plots()
