from .disk import Disk
from .dram import DRAM
from .gpu import GPU
from .link import Link

# Define common GPU hardware configurations
GPUS = {
    # H200 SXM: 3360 GB/s HBM, 141 GB HBM capacity, 4000 TFLOPS peak FP
    "H200": GPU(name="H200 SXM", hbm_bw=3360e9, hbm_capacity=141e9, peak_flops=4000e12),
    # H100 NVL: 3360 GB/s HBM, 80 GB HBM capacity, 4000 TFLOPS peak FP
    "H100": GPU(name="H100 NVL", hbm_bw=3360e9, hbm_capacity=80e9, peak_flops=1979e12),
    # A100 80GB: 2039 GB/s HBM, 80 GB HBM capacity, 312 TFLOPS peak FP
    "A100": GPU(name="A100 80GB", hbm_bw=2039e9, hbm_capacity=80e9, peak_flops=312e12),
    # RTX 6000 Ada: 960 GB/s HBM, 48 GB HBM capacity, 182 TFLOPS peak FP
    "RTX6000": GPU(name="RTX 6000 Ada", hbm_bw=960e9, hbm_capacity=48e9, peak_flops=182e12),
    # Tesla V100: 897 GB/s HBM, 32 GB HBM capacity, 125 TFLOPS peak FP
    "V100": GPU(name="Tesla V100", hbm_bw=897e9, hbm_capacity=32e9, peak_flops=125e12),
    # RTX A5000: 768 GB/s HBM, 24 GB HBM capacity, 55 TFLOPS peak FP
    "A5000": GPU(name="RTX A5000", hbm_bw=768e9, hbm_capacity=24e9, peak_flops=55e12),
}

# Define common DRAM configurations
DRAMS = {
    "DDR5-6000": DRAM(
        name="DDR5-6000", bandwidth=85e9, capacity=32e9
    ),  # 85 GB/s, 32 GB
    "DDR5-7200": DRAM(
        name="DDR5-7200", bandwidth=81e9, capacity=32e9
    ),  # 81 GB/s, 32 GB
    "DDR5-5600": DRAM(
        name="DDR5-5600", bandwidth=62e9, capacity=32e9
    ),  # 62 GB/s, 32 GB
    "DDR4-3200": DRAM(
        name="DDR4-3200", bandwidth=30e9, capacity=16e9
    ),  # 30 GB/s, 16 GB
    "DDR4-2133": DRAM(
        name="DDR4-2133", bandwidth=25e9, capacity=16e9
    ),  # 25 GB/s, 16 GB
    "DDR3-1600": DRAM(
        name="DDR3-1600", bandwidth=20e9, capacity=8e9
    ),  # 20 GB/s, 8 GB
}

# Define common disk configurations
DISKS = {
    "HDD": Disk(
        name="HDD", bandwidth=100e6, capacity=4e12
    ),  # 100 MB/s, 4 TB
    "X110": Disk(
        name="SSD X110 SATA M.2", bandwidth=300e6, capacity=256e9
    ),  # 300 MB/s, 256 GB
    "M550": Disk(
        name="SSD M550 SATA M.2", bandwidth=400e6, capacity=512e9
    ),  # 400 MB/s, 512 GB
    "NVMe980": Disk(
        name="SSD NVMe 980 M.2", bandwidth=2e9, capacity=2e12
    ),  # 2 GB/s, 2 TB
    "NVMeT700": Disk(
        name="SSD NVMe T700 M.2", bandwidth=5e9, capacity=4e12
    ),  # 5 GB/s, 4 TB
    "NVMeT700R0": Disk(
        name="SSD NVMe T700 M.2 (Raid 0)", bandwidth=10e9, capacity=8e12
    ),  # 10 GB/s, 8 TB
    "NVMeT700R5": Disk(
        name="SSD NVMe T700 M.2 (Raid 5)", bandwidth=20e9, capacity=16e12
    ),  # 20 GB/s, 16 TB
}

# Define common link configurations
LINKS = {
    "PCIe3": Link(name="PCIe 3.0 x16", bandwidth=16e9),     # 16 GB/s
    "PCIe4": Link(name="PCIe 4.0 x16", bandwidth=32e9),     # 32 GB/s
    "PCIe5": Link(name="PCIe 5.0 x16", bandwidth=64e9),     # 64 GB/s
    "NVLink3": Link(name="NVLink 3.0", bandwidth=600e9),    # 600 GB/s
    "NVLink4": Link(name="NVLink 4.0", bandwidth=900e9),    # 900 GB/s
    "NVLink5": Link(name="NVLink 5.0", bandwidth=1800e9),   # 1800 GB/s
    "NVLink6": Link(name="NVLink 6.0", bandwidth=3600e9),   # 3600 GB/s
}
