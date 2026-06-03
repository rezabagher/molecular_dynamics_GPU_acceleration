"""
cpu_forces.py — CPU reference implementation
=============================================
Pairwise force and potential energy using plain NumPy.

TWO potentials are available (selectable via `potential_type`):

  'soft' (default) — Soft harmonic repulsion:
      U(r) = A·(1 − r/r₀)²   for  r < r₀,  else  0

      Simple, always finite, never diverges.  Ideal for learning,
      stable with any reasonable time-step.

  'lj'  — Lennard-Jones (classic MD potential):
      U(r) = 4ε [(σ/r)¹² − (σ/r)⁶],  truncated at r_cut = 2.5σ

      Physically rich (gas, liquid, solid phases), but requires
      a small time-step (dt ≤ 0.005) and gentle initialisation.

Force formula for soft potential (unit-normalised direction vector):
    F_ij = −∇U = (2A/r₀) · (1 − r/r₀) · r̂_ij   (outward = repulsive)

Force formula for LJ (reduced units, ε = σ = 1):
    F_ij = (48/r²) [(1/r)¹² − 0.5·(1/r)⁶] · Δr_ij
"""

import numpy as np


def compute_forces_cpu(
    positions:      np.ndarray,
    box_size:       float,
    # --- shared params -------------------------------------------------
    potential_type: str   = "soft",
    # --- soft potential params ------------------------------------------
    r0:             float = 2.0,     # interaction diameter
    A:              float = 50.0,    # repulsion strength
    # --- LJ params ------------------------------------------------------
    r_cut:          float = 2.5,
    epsilon:        float = 1.0,
    sigma:          float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Compute all pairwise forces and total potential energy on the CPU.

    Parameters
    ----------
    positions      : ndarray (N, 2), float32
    box_size       : float  — periodic box side length
    potential_type : 'soft' (default) or 'lj'
    r0             : float  — soft potential range (soft only)
    A              : float  — soft potential strength (soft only)
    r_cut          : float  — LJ outer cut-off (lj only)
    epsilon, sigma : float  — LJ parameters in reduced units (lj only)

    Returns
    -------
    forces    : ndarray (N, 2), float32
    potential : float
    """
    n = positions.shape[0]
    forces    = np.zeros((n, 2), dtype=np.float32)
    potential = 0.0

    if potential_type == "soft":
        return _soft_forces(positions, box_size, n, forces, r0, A)
    else:
        return _lj_forces(positions, box_size, n, forces, r_cut, epsilon, sigma)


# ── soft potential ────────────────────────────────────────────────────────────

def _soft_forces(positions, box_size, n, forces, r0, A):
    """Harmonic repulsion: U = A*(1 - r/r0)^2 for r < r0."""
    potential = 0.0
    r0sq = r0 * r0

    for i in range(n):
        for j in range(i + 1, n):
            dx = float(positions[j, 0] - positions[i, 0])
            dy = float(positions[j, 1] - positions[i, 1])
            # minimum-image convention (periodic boundary)
            dx -= box_size * round(dx / box_size)
            dy -= box_size * round(dy / box_size)

            r2 = dx * dx + dy * dy

            if r2 >= r0sq or r2 < 1e-14:
                continue

            r       = r2 ** 0.5
            overlap = 1.0 - r / r0              # ∈ (0, 1] when r < r0

            potential += A * overlap * overlap   # U = A*(1 - r/r0)²

            # |F| = dU/dr = 2A*(1 - r/r0)/r0
            # F_vec = |F| * r̂  (outward from j→i = repulsive)
            f_mag = 2.0 * A * overlap / (r0 * r)
            fx = f_mag * dx
            fy = f_mag * dy

            forces[i, 0] -= fx;  forces[i, 1] -= fy   # repel i away from j
            forces[j, 0] += fx;  forces[j, 1] += fy   # repel j away from i

    return forces, potential


# ── Lennard-Jones potential ────────────────────────────────────────────────────

def _lj_forces(positions, box_size, n, forces, r_cut, epsilon, sigma):
    """Standard LJ potential: U = 4ε[(σ/r)¹² - (σ/r)⁶], cut at r_cut."""
    potential = 0.0
    r_cut2 = r_cut * r_cut
    s2     = sigma * sigma

    for i in range(n):
        for j in range(i + 1, n):
            dx = float(positions[j, 0] - positions[i, 0])
            dy = float(positions[j, 1] - positions[i, 1])
            dx -= box_size * round(dx / box_size)
            dy -= box_size * round(dy / box_size)

            r2 = dx * dx + dy * dy

            if r2 > r_cut2 or r2 < 1e-14:
                continue

            sr2  = s2 / r2
            sr6  = sr2 * sr2 * sr2
            sr12 = sr6 * sr6

            potential += 4.0 * epsilon * (sr12 - sr6)

            # F_scalar = (48ε/r²)[(σ/r)^12 - 0.5*(σ/r)^6]
            f_scalar = (48.0 * epsilon / r2) * (sr12 - 0.5 * sr6)
            forces[i, 0] += f_scalar * dx;  forces[i, 1] += f_scalar * dy
            forces[j, 0] -= f_scalar * dx;  forces[j, 1] -= f_scalar * dy

    return forces, potential
