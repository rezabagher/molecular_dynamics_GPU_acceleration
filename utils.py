"""
utils.py — Helper functions
============================
Particle initialisation and small numeric utilities shared by
both the CPU and GPU code paths.
"""

import numpy as np


def init_particles(n_particles: int, box_size: float, seed: int = 42,
                   target_temp: float = 1.0):
    """
    Place N particles on a jittered grid and give them Maxwell-Boltzmann
    velocities scaled to target_temp.

    The grid placement prevents particles from starting too close together
    (which causes catastrophically large LJ repulsions on step 1).

    Parameters
    ----------
    n_particles  : int    Number of particles.
    box_size     : float  Side length of the square simulation box.
    seed         : int    Random seed for reproducibility.
    target_temp  : float  Rescale velocities to this temperature.

    Returns
    -------
    positions  : ndarray, shape (N, 2), float32
    velocities : ndarray, shape (N, 2), float32
    """
    rng = np.random.default_rng(seed)

    # --- positions: square grid (guaranteed minimum spacing) ----------------
    cols    = int(np.ceil(np.sqrt(n_particles)))
    rows    = int(np.ceil(n_particles / cols))
    spacing = box_size / max(cols, rows)

    xs = (np.arange(cols) + 0.5) * spacing
    ys = (np.arange(rows) + 0.5) * spacing
    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])[:n_particles]

    # tiny jitter: no more than 10 % of spacing so particles stay well apart
    jitter = rng.uniform(-spacing * 0.1, spacing * 0.1, size=grid.shape)
    positions = np.clip(grid + jitter, 0.1, box_size - 0.1).astype(np.float32)

    # --- velocities: zero-mean, rescaled to target_temp ---------------------
    velocities = rng.standard_normal(size=(n_particles, 2)).astype(np.float32)
    velocities -= velocities.mean(axis=0)           # zero net momentum

    # rescale to desired temperature
    t_now = temperature(velocities)
    if t_now > 0:
        velocities *= np.sqrt(target_temp / t_now)

    return positions, velocities


def kinetic_energy(velocities: np.ndarray, mass: float = 1.0) -> float:
    """KE = 0.5 * m * sum(v²)"""
    return 0.5 * mass * float(np.sum(velocities ** 2))


def temperature(velocities: np.ndarray, mass: float = 1.0) -> float:
    """
    Instantaneous temperature from the equipartition theorem.
    T = 2*KE / (N * dim - dim)   [2-D, fixed COM removed]
    """
    n, dim = velocities.shape
    ke = kinetic_energy(velocities, mass)
    dof = n * dim - dim
    return 2.0 * ke / dof if dof > 0 else 0.0
