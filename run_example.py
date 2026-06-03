"""
run_example.py — Minimal working example
=========================================
Run from the repository root:

    python run_example.py                    # 256 particles, soft potential, CPU
    python run_example.py --gpu              # GPU (requires CUDA + Numba)
    python run_example.py --lj              # Lennard-Jones NVT (CPU, thermostatted)
    python run_example.py --n 512           # custom particle count
    python run_example.py --benchmark       # CPU vs GPU timing table
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from md_sim.simulation import Simulation
from md_sim.gpu_forces  import _CUDA_AVAILABLE


def main():
    parser = argparse.ArgumentParser(description="MD-Sim toy simulator")
    parser.add_argument("--gpu",       action="store_true",
                        help="Use GPU acceleration (requires CUDA + Numba)")
    parser.add_argument("--lj",        action="store_true",
                        help="Use Lennard-Jones potential (CPU NVT, thermostatted)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Print CPU vs GPU performance table")
    parser.add_argument("--n",     type=int, default=256,
                        help="Number of particles (default: 256)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Production run steps (default: 500)")
    args = parser.parse_args()

    # ── benchmark ─────────────────────────────────────────────────────────────
    if args.benchmark:
        from md_sim.benchmark import run_benchmark
        run_benchmark()
        return

    # ── simulation ────────────────────────────────────────────────────────────
    use_gpu        = args.gpu
    potential_type = "lj" if args.lj else "soft"

    if use_gpu and not _CUDA_AVAILABLE:
        print("Warning: CUDA not available — falling back to CPU.")
        use_gpu = False
    if args.lj and use_gpu:
        print("Note: LJ potential runs on CPU only.")
        use_gpu = False

    if potential_type == "lj":
        # LJ: use NVT (thermostat always on), low density gas
        import numpy as np
        dt          = 0.001
        equil_steps = 1000
        n           = min(args.n, 64)   # keep LJ manageable
        box_size    = float(np.sqrt(n / 0.05))   # very low density
        print(f"[LJ mode] N={n}  box={box_size:.1f}  rho=0.05  dt={dt}")
        print("          LJ runs in NVT (thermostat always ON)")
    else:
        dt          = 0.01
        equil_steps = 300
        n           = args.n
        box_size    = 25.0

    sim = Simulation(
        n_particles    = n,
        box_size       = box_size,
        dt             = dt,
        use_gpu        = use_gpu,
        potential_type = potential_type,
        target_temp    = 1.0,
    )

    sim.equilibrate(steps=equil_steps)

    # LJ: keep thermostat on for stability (NVT ensemble)
    # Soft: NVE production (thermostat off)
    if potential_type == "lj":
        print(f"\nProduction run ({args.steps} steps, thermostat ON) …")
        log_every = max(1, args.steps // 10)
        from md_sim.utils import kinetic_energy, temperature
        import numpy as np
        print(f"\n{'Step':>6}  {'Time':>8}  {'KE':>10}  {'PE':>10}  {'Temp':>8}")
        print("-" * 50)
        for s in range(args.steps):
            state = sim.step(thermostat=True)
            if (s + 1) % log_every == 0 or s == 0:
                print(f"{state.step:>6}  {state.time:8.3f}  "
                      f"{state.ke:10.4f}  {state.pe:10.4f}  {state.temp:8.4f}")
    else:
        sim.run(steps=args.steps, log_every=max(1, args.steps // 10))


if __name__ == "__main__":
    main()
