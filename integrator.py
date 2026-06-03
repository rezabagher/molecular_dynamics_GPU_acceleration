"""
integrator.py — Velocity-Verlet integrator
===========================================
The Velocity-Verlet algorithm is the standard choice for MD simulations
because it is time-reversible, symplectic (conserves a shadow Hamiltonian),
and second-order accurate with only one force evaluation per step.

Algorithm (one timestep Δt)
---------------------------
  1.  x(t + Δt)  =  x(t) + v(t)·Δt + 0.5·a(t)·Δt²
  2.  Compute F(t + Δt)  →  a(t + Δt) = F/m
  3.  v(t + Δt)  =  v(t) + 0.5·[a(t) + a(t + Δt)]·Δt

Step 2 is delegated to the caller (simulation.py) so this module stays
force-agnostic and can be swapped for other integrators.
"""

import numpy as np


def velocity_verlet_step1(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    dt: float,
    box_size: float,
    mass: float = 1.0,
) -> np.ndarray:
    """
    **First half** of Velocity-Verlet: advance positions and half-kick velocities.

    After this call:
      - positions are at t + Δt  (wrap applied for periodic boundaries)
      - velocities hold a *half-step* intermediate value (not yet physical)

    Parameters
    ----------
    positions  : ndarray (N, 2)  — current positions, modified IN PLACE
    velocities : ndarray (N, 2)  — current velocities, modified IN PLACE
    forces     : ndarray (N, 2)  — forces at time t
    dt         : float           — timestep
    box_size   : float           — periodic box side length
    mass       : float           — particle mass (reduced units: 1.0)

    Returns
    -------
    positions  : ndarray (N, 2)  — same array, updated in place (returned for clarity)
    """
    acceleration = forces / mass                        # a(t) = F(t) / m

    # --- position update: x(t+Δt) = x(t) + v(t)·Δt + 0.5·a(t)·Δt² --------
    positions += velocities * dt + 0.5 * acceleration * dt ** 2

    # --- periodic boundary condition (wrap) ----------------------------------
    positions %= box_size

    # --- half-velocity update: v* = v(t) + 0.5·a(t)·Δt ---------------------
    velocities += 0.5 * acceleration * dt

    return positions


def velocity_verlet_step2(
    velocities: np.ndarray,
    forces_new: np.ndarray,
    dt: float,
    mass: float = 1.0,
) -> np.ndarray:
    """
    **Second half** of Velocity-Verlet: complete velocity update with new forces.

    Call this after computing forces at the new positions.

    Parameters
    ----------
    velocities  : ndarray (N, 2)  — half-step velocities from step1, updated IN PLACE
    forces_new  : ndarray (N, 2)  — forces at t + Δt (just computed)
    dt          : float           — timestep
    mass        : float           — particle mass

    Returns
    -------
    velocities  : ndarray (N, 2)  — full velocities at t + Δt
    """
    acceleration_new = forces_new / mass

    # --- complete velocity: v(t+Δt) = v* + 0.5·a(t+Δt)·Δt -----------------
    velocities += 0.5 * acceleration_new * dt

    return velocities
