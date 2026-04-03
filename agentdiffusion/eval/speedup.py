"""Speedup benchmarking: compare wall-clock time of diffusion vs traditional ABM."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class SpeedupResult:
    agent_count: int
    num_steps: int
    diffusion_time_s: float
    diffusion_gpu_mem_mb: float
    baseline_time_s: float | None
    speedup_ratio: float | None

    @property
    def summary(self) -> str:
        ratio_str = f"{self.speedup_ratio:.1f}x" if self.speedup_ratio else "N/A"
        return (
            f"Agents={self.agent_count}, Steps={self.num_steps}: "
            f"Diffusion={self.diffusion_time_s:.2f}s, "
            f"GPU={self.diffusion_gpu_mem_mb:.0f}MB, "
            f"Speedup={ratio_str}"
        )


def benchmark_diffusion(
    inference_fn: callable,
    initial_state: torch.Tensor,   # [1, H, W, C]
    num_steps: int,
    warmup_runs: int = 3,
    measure_runs: int = 5,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Benchmark diffusion inference.

    Returns: (avg_time_seconds, peak_gpu_memory_mb)
    """
    if device is None:
        device = initial_state.device

    # Warmup
    for _ in range(warmup_runs):
        _ = inference_fn(initial_state, num_steps)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(measure_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        _ = inference_fn(initial_state, num_steps)

        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    avg_time = sum(times) / len(times)

    gpu_mem = 0.0
    if device.type == "cuda":
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return avg_time, gpu_mem


def run_speedup_benchmark(
    inference_fn: callable,
    agent_counts: list[int] = [10_000, 100_000],
    num_steps_list: list[int] = [10, 50, 100],
    raw_dim: int = 128,
    baseline_times: dict[tuple[int, int], float] | None = None,
) -> list[SpeedupResult]:
    """Run comprehensive speedup benchmark across scales."""
    import math
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for n_agents in agent_counts:
        H = int(math.ceil(math.sqrt(n_agents)))
        W = int(math.ceil(n_agents / H))

        initial_state = torch.randn(1, H, W, raw_dim, device=device)

        for n_steps in num_steps_list:
            try:
                diff_time, gpu_mem = benchmark_diffusion(
                    inference_fn, initial_state, n_steps,
                    warmup_runs=2, measure_runs=3, device=device,
                )
            except torch.cuda.OutOfMemoryError:
                diff_time, gpu_mem = float("inf"), float("inf")

            baseline = None
            speedup = None
            if baseline_times and (n_agents, n_steps) in baseline_times:
                baseline = baseline_times[(n_agents, n_steps)]
                speedup = baseline / diff_time if diff_time > 0 else float("inf")

            results.append(SpeedupResult(
                agent_count=n_agents,
                num_steps=n_steps,
                diffusion_time_s=diff_time,
                diffusion_gpu_mem_mb=gpu_mem,
                baseline_time_s=baseline,
                speedup_ratio=speedup,
            ))

    return results
