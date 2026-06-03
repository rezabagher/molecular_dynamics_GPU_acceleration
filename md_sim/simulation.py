"""
simulation.py — Main simulation driver
=======================================
Ties together initialisation, force computation (CPU or GPU),
and the Velocity-Verlet integrator into a clean run loop.

By default uses the stable soft-repulsion potential.  Switch to
full Lennard-Jones with potential_type='lj' (CPU only; requires
smaller dt ≈ 0.001 and a long equilibration).

Example
-------
    from md_sim.simulation import Simulation

    # Soft repulsion (stable, default)
    sim = Simulation(n_particles=256, box_size=25.0, dt=0.01)
    sim.run(steps=500, log_every=50)

    # Full Lennard-Jones (CPU only)
    sim = Simulation(n_particles=64, box_size=15.0, dt=0.001,
                     potential_type='lj')
    sim.equilibrate(steps=2000)
    sim.run(steps=500)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .utils      import init_particles, kinetic_energy, temperature
from .cpu_forces import compute_forces_cpu
from .gpu_forces import compute_forces_gpu, _CUDA_AVAILABLE
from .integrator import velocity_verlet_step1, velocity_verlet_step2


@dataclass
class SimulationState:
    """Snapshot of one logged timestep."""
    step:    int
    time:    float
    ke:      float
    pe:      float
    total_e: float
    temp:    float


class Simulation:
    """
    2-D particle simulation with CPU or GPU acceleration.

    Parameters
    ----------
    n_particles    : Number of particles.
    box_size       : Periodic box side length.
    dt             : Integration timestep.
                     Soft potential: dt = 0.01 is comfortable.
                     LJ potential:   dt ≤ 0.001 recommended.
    use_gpu        : Use Numba CUDA kernel if True (soft potential only).
    potential_type : 'soft' (default, stable) or 'lj' (Lennard-Jones).
    target_temp    : Initial temperature; thermostat target.
    r0             : Soft-potential interaction diameter.
    A              : Soft-potential repulsion strength.
    r_cut          : LJ cut-off radius (lj mode only).
    seed           : Random seed.
    """

    def __init__(
        self,
        n_particles:    int   = 256,
        box_size:       float = 25.0,
        dt:             float = 0.01,
        use_gpu:        bool  = False,
        potential_type: str   = "soft",
        target_temp:    float = 1.0,
        r0:             float = 2.0,
        A:              float = 50.0,
        r_cut:          float = 2.5,
        seed:           int   = 42,
    ):
        self.n_particles    = n_particles
        self.box_size       = box_size
        self.dt             = dt
        self.potential_type = potential_type
        self.target_temp    = target_temp
        self.r0             = r0
        self.A              = A
        self.r_cut          = r_cut

        self.use_gpu = use_gpu and _CUDA_AVAILABLE
        if use_gpu and not _CUDA_AVAILABLE:
            print("[Warning] CUDA not available — falling back to CPU.")
        if use_gpu and potential_type == "lj":
            print("[Warning] GPU kernel only supports 'soft' potential; using CPU for LJ.")
            self.use_gpu = False

        self.positions, self.velocities = init_particles(
            n_particles, box_size, seed, target_temp
        )
        self.forces = np.zeros((n_particles, 2), dtype=np.float32)
        self.forces, self._pe = self._compute_forces()

        self._step = 0
        self._time = 0.0
        self.history: list[SimulationState] = []

        device = "GPU (CUDA)" if self.use_gpu else "CPU"
        print(
            f"Simulation ready: {n_particles} particles | "
            f"box={box_size:.1f} | dt={dt} | "
            f"potential={potential_type} | T={target_temp} | {device}"
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _compute_forces(self):
        if self.use_gpu:
            forces, _ = compute_forces_gpu(
                self.positions, self.box_size, self.r0, self.A
            )
            return forces, float("nan")
        return compute_forces_cpu(
            self.positions, self.box_size,
            potential_type=self.potential_type,
            r0=self.r0, A=self.A, r_cut=self.r_cut,
        )

    def _rescale_velocities(self):
        """Velocity-rescaling thermostat: T → target_temp."""
        t_now = temperature(self.velocities)
        if t_now > 1e-10:
            self.velocities *= np.sqrt(self.target_temp / t_now)

    def _log(self) -> SimulationState:
        ke   = kinetic_energy(self.velocities)
        pe   = self._pe
        te   = ke + (pe if not np.isnan(pe) else 0.0)
        temp = temperature(self.velocities)
        return SimulationState(self._step, self._time, ke, pe, te, temp)

    # ── public API ────────────────────────────────────────────────────────────

    def step(self, thermostat: bool = False) -> SimulationState:
        """Advance one Velocity-Verlet timestep."""
        velocity_verlet_step1(
            self.positions, self.velocities, self.forces,
            self.dt, self.box_size,
        )
        self.forces, self._pe = self._compute_forces()
        velocity_verlet_step2(self.velocities, self.forces, self.dt)

        if thermostat:
            self._rescale_velocities()

        self._step += 1
        self._time += self.dt
        return self._log()

    def equilibrate(self, steps: int = 300, verbose: bool = True):
        """
        Run *steps* thermostatted steps to reach thermal equilibrium.

        The thermostat rescales velocities every step, keeping T = target_temp.
        Call this before a production run for physically meaningful results.
        """
        if verbose:
            print(f"\nEquilibrating {steps} steps (thermostat ON) …")
        for _ in range(steps):
            self.step(thermostat=True)
        s = self._log()
        if verbose:
            print(f"  Done.  T = {s.temp:.4f}  KE = {s.ke:.4f}")

    def run(
        self,
        steps:     int  = 500,
        log_every: int  = 50,
        verbose:   bool = True,
    ) -> list[SimulationState]:
        """
        Production run in the NVE ensemble (no thermostat).

        Parameters
        ----------
        steps     : total integration steps
        log_every : record and print a snapshot every N steps
        verbose   : print progress table

        Returns
        -------
        list of SimulationState snapshots
        """
        self.history.clear()
        t0 = time.perf_counter()

        if verbose:
            pe_col = "PE" if not self.use_gpu else "PE(N/A)"
            print(
                f"\n{'Step':>6}  {'Time':>8}  {'KE':>10}  "
                f"{pe_col:>10}  {'Temp':>8}"
            )
            print("-" * 50)

        for s in range(steps):
            state = self.step(thermostat=False)

            if (s + 1) % log_every == 0 or s == 0:
                self.history.append(state)
                if verbose:
                    pe_s = (f"{state.pe:10.4f}" if not np.isnan(state.pe)
                            else "       N/A")
                    print(
                        f"{state.step:>6}  {state.time:8.3f}  "
                        f"{state.ke:10.4f}  {pe_s}  {state.temp:8.4f}"
                    )

        elapsed = time.perf_counter() - t0
        if verbose:
            print(
                f"\nDone.  {steps} steps in {elapsed:.3f} s "
                f"({steps / elapsed:.1f} steps/s)"
            )
        return self.history
