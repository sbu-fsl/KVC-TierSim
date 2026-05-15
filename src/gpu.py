from dataclasses import dataclass

@dataclass(frozen=True)
class GPU:
    name: str
    hbm_bandwidth: int
    hbm_capacity: int
    peak_flops: int

    def gpu_compute_band(self, model_params: int, gpu_count: int = 1):
        # compute time per token
        flops_per_token = (
            2 * model_params
        )  # 2 FLOPs per parameter (1 for forward pass, 1 for backward pass)
        t_compute = flops_per_token / (self.peak_flops * 0.25)  # assume 25% efficiency

        return t_compute / gpu_count
