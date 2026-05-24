from .disk import Disk
from .dram import DRAM
from .gpu import GPU
from .link import Link

# Define common GPU hardware configurations
GPUS = {
    # H200 SXM: 3360 GB/s HBM, 141 GB HBM capacity, 4000 TFLOPS peak FP
    "H200": GPU(name="H200", hbm_bandwidth=3360e9, hbm_capacity=141e9, peak_flops=4000e12, cost=40_000),
    # H100 NVL: 3360 GB/s HBM, 80 GB HBM capacity, 4000 TFLOPS peak FP
    "H100": GPU(name="H100", hbm_bandwidth=3360e9, hbm_capacity=80e9, peak_flops=1979e12, cost=24_500),
    # A100 80GB: 2039 GB/s HBM, 80 GB HBM capacity, 312 TFLOPS peak FP
    "A100": GPU(name="A100", hbm_bandwidth=2039e9, hbm_capacity=80e9, peak_flops=312e12, cost=12_000),
    # RTX 6000 Ada: 960 GB/s HBM, 48 GB HBM capacity, 182 TFLOPS peak FP
    "RTX6000": GPU(name="RTX 6000 Ada", hbm_bandwidth=960e9, hbm_capacity=48e9, peak_flops=182e12, cost=8_000),
    # Tesla V100: 897 GB/s HBM, 32 GB HBM capacity, 125 TFLOPS peak FP
    "V100": GPU(name="Tesla V100", hbm_bandwidth=897e9, hbm_capacity=32e9, peak_flops=125e12, cost=3_000),
    # RTX A5000: 768 GB/s HBM, 24 GB HBM capacity, 55 TFLOPS peak FP
    "A5000": GPU(name="RTX A5000", hbm_bandwidth=768e9, hbm_capacity=24e9, peak_flops=55e12, cost=2_300),

    # Grouping by generation
    "High-End": GPU(name="High-End", hbm_bandwidth=3360e9, hbm_capacity=141e9, peak_flops=4000e12, cost=40_000),
    "Mid-Range": GPU(name="Mid-Range", hbm_bandwidth=2039e9, hbm_capacity=80e9, peak_flops=312e12, cost=12_000),
}

# Define common DRAM configurations
DRAMS = {
    "DDR5-6000": DRAM(
        name="DDR5-6000", bandwidth=85e9, capacity=32e9, cost=500
    ),  # 85 GB/s, 32 GB
    "DDR5-7200": DRAM(
        name="DDR5-7200", bandwidth=81e9, capacity=32e9, cost=490
    ),  # 81 GB/s, 32 GB
    "DDR5-5600": DRAM(
        name="DDR5-5600", bandwidth=62e9, capacity=32e9, cost=280
    ),  # 62 GB/s, 32 GB
    "DDR4-3200": DRAM(
        name="DDR4-3200", bandwidth=30e9, capacity=16e9, cost=150
    ),  # 30 GB/s, 16 GB
    "DDR4-2133": DRAM(
        name="DDR4-2133", bandwidth=25e9, capacity=16e9, cost=120
    ),  # 25 GB/s, 16 GB
    "DDR3-1600": DRAM(
        name="DDR3-1600", bandwidth=20e9, capacity=8e9, cost=60
    ),  # 20 GB/s, 8 GB

    # Grouping by generation
    "DDR5": DRAM(name="DDR5", bandwidth=85e9, capacity=32e9, cost=500),  # 85 GB/s, 32 GB
    "DDR4": DRAM(name="DDR4", bandwidth=30e9, capacity=16e9, cost=150),  # 30 GB/s, 16 GB
}

# Define common disk configurations
DISKS = {
    "HDD": Disk(
        name="HDD", bandwidth=100e6, capacity=4e12, cost=100
    ),  # 100 MB/s, 4 TB
    "X110": Disk(
        name="SSD X110 SATA M.2", bandwidth=300e6, capacity=2e12, cost=200
    ),  # 300 MB/s, 2 TB
    "M550": Disk(
        name="SSD M550 SATA M.2", bandwidth=400e6, capacity=512e9, cost=150
    ),  # 400 MB/s, 512 GB
    "NVMe980": Disk(
        name="SSD NVMe 980 M.2", bandwidth=2e9, capacity=2e12, cost=300
    ),  # 2 GB/s, 2 TB
    "NVMeT700": Disk(
        name="SSD NVMe T700 M.2", bandwidth=5e9, capacity=4e12, cost=500
    ),  # 5 GB/s, 4 TB
    "NVMeT700R0": Disk(
        name="SSD NVMe T700 M.2 (Raid 0)", bandwidth=10e9, capacity=8e12, cost=1000
    ),  # 10 GB/s, 8 TB
    "NVMeT700R5": Disk(
        name="SSD NVMe T700 M.2 (Raid 5)", bandwidth=20e9, capacity=16e12, cost=2000
    ),  # 20 GB/s, 16 TB

    # Grouping by type
    "SATA": Disk(name="SATA SSD", bandwidth=300e6, capacity=2e12, cost=200),   # 300 MB/s, 2 TB
    "NVMe": Disk(name="NVMe SSD", bandwidth=2e9, capacity=2e12, cost=500),     # 2 GB/s, 2 TB
}

# Define common link configurations
LINKS = {
    "PCIe3": Link(name="PCIe 3.0 x16", bandwidth=16e9, cost=15),     # 16 GB/s
    "PCIe4": Link(name="PCIe 4.0 x16", bandwidth=32e9, cost=90),     # 32 GB/s
    "PCIe5": Link(name="PCIe 5.0 x16", bandwidth=64e9, cost=200),     # 64 GB/s
    "NVLink3": Link(name="NVLink 3.0", bandwidth=600e9, cost=150),    # 600 GB/s
    "NVLink4": Link(name="NVLink 4.0", bandwidth=900e9, cost=800),    # 900 GB/s
    "NVLink5": Link(name="NVLink 5.0", bandwidth=1800e9, cost=1600),   # 1800 GB/s
    "NVLink6": Link(name="NVLink 6.0", bandwidth=3600e9, cost=3200),   # 3600 GB/s

    # Grouping by type
    "PCIe": Link(name="PCIe", bandwidth=32e9, cost=90),       # 32 GB/s
    "NVLink": Link(name="NVLink", bandwidth=600e9, cost=150),  # 600 GB/s
}
