from dataclasses import dataclass

@dataclass(frozen=True)
class GPU:
    name: str
    hbm_bandwidth: int
    hbm_capacity: int
    peak_flops: int
    cost: int

    def gpu_compute_band(self, model_params: int, gpu_count: int = 1, eta: float = 0.6):
        flops_per_token = 2 * model_params
        total_flops = self.peak_flops * gpu_count * eta
        t_compute = flops_per_token / total_flops

        return t_compute
