"""
benchmark.py — CPU vs GPU performance comparison
=================================================
Measures wall-clock time for force-computation calls at several
particle counts and prints a formatted comparison table.

Usage (from the repo root)
--------------------------
    python -m md_sim.benchmark
    python run_example.py --benchmark

What is measured
----------------
Each trial calls the force kernel REPEATS times and reports:
  • Mean time per call (ms)
  • Standard deviation (ms)
  • GPU speedup over CPU (×)

GPU timings include host↔device memory transfers (the realistic case).
cuda.synchronize() is called inside the GPU wrapper, so timing is exact.
"""

import time
import textwrap

import numpy as np

from .cpu_forces import compute_forces_cpu
from .gpu_forces import compute_forces_gpu, _CUDA_AVAILABLE
from .utils      import init_particles


def _time_fn(fn, repeats: int = 5) -> tuple[float, float]:
    """
    Call fn() `repeats + 1` times.  Discard the first run (warm-up /
    JIT compilation) and return (mean_ms, std_ms) over the rest.
    """
    times = []
    for i in range(repeats + 1):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        if i > 0:
            times.append((t1 - t0) * 1_000.0)   # seconds → ms
    return float(np.mean(times)), float(np.std(times))


def run_benchmark(
    particle_counts: list[int] = (64, 128, 256, 512, 1024),
    box_size:        float      = 25.0,
    r0:              float      = 2.0,
    A:               float      = 50.0,
    repeats:         int        = 5,
):
    """
    Benchmark CPU and (optionally) GPU force kernels for various N.

    Parameters
    ----------
    particle_counts : sequence of particle counts to benchmark
    box_size        : simulation box side length
    r0, A           : soft potential parameters
    repeats         : timing repetitions after one warm-up
    """
    print("\n" + "=" * 67)
    print("  Molecular Dynamics — CPU vs GPU Force Kernel Benchmark")
    print("=" * 67)

    if _CUDA_AVAILABLE:
        try:
            from numba import cuda
            device = cuda.get_current_device()
            print(f"  GPU     : {device.name}")
        except Exception:
            print("  GPU     : CUDA device detected")
    else:
        print("  GPU     : NOT AVAILABLE — install Numba + CUDA toolkit")

    print(f"  Repeats : {repeats}  (first call is warm-up, discarded)")
    print(f"  Potential: soft repulsion  r0={r0}  A={A}")
    print()

    col = 14
    header = f"{'N':>6}  {'CPU mean (ms)':>{col}}  {'CPU std':>8}"
    if _CUDA_AVAILABLE:
        header += f"  {'GPU mean (ms)':>{col}}  {'GPU std':>8}  {'Speedup':>8}"
    print(header)
    print("-" * (len(header) + 2))

    for n in particle_counts:
        pos, _ = init_particles(n, box_size, seed=0)

        # CPU
        cpu_fn               = lambda: compute_forces_cpu(pos, box_size, r0=r0, A=A)
        cpu_mean, cpu_std    = _time_fn(cpu_fn, repeats)

        row = f"{n:>6}  {cpu_mean:>{col}.2f}  {cpu_std:>8.2f}"

        # GPU (optional)
        if _CUDA_AVAILABLE:
            gpu_fn            = lambda: compute_forces_gpu(pos, box_size, r0=r0, A=A)
            gpu_mean, gpu_std = _time_fn(gpu_fn, repeats)
            speedup           = cpu_mean / gpu_mean if gpu_mean > 0 else float("inf")
            row += f"  {gpu_mean:>{col}.2f}  {gpu_std:>8.2f}  {speedup:>7.1f}×"

        print(row)

    print()
    print(textwrap.dedent("""
    Notes
    -----
    • GPU timings include host→device and device→host memory transfers.
    • The CPU kernel is pure Python (O(N²) loops) — intentionally
      readable, not optimised.  A NumPy-vectorised version would be
      significantly faster.
    • GPU speedup typically becomes substantial at N ≥ 512.
    • Run on a machine with a discrete NVIDIA GPU for meaningful numbers.
    """).strip())


if __name__ == "__main__":
    run_benchmark()
