"""
tests/test_md_sim.py — Pytest test suite for md_sim
=====================================================
Covers:
  - utils.py       : init_particles, kinetic_energy, temperature
  - cpu_forces.py  : soft and Lennard-Jones potentials
  - integrator.py  : velocity_verlet_step1, velocity_verlet_step2
  - simulation.py  : Simulation (initialisation, equilibration, run)

Run with:
    pytest tests/ -v
    pytest tests/ -v --tb=short   # shorter tracebacks
"""

import numpy as np
import pytest

# ── imports ──────────────────────────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from md_sim.utils import init_particles, kinetic_energy, temperature
from md_sim.cpu_forces import compute_forces_cpu
from md_sim.gpu_forces import _CUDA_AVAILABLE, compute_forces_gpu
from md_sim.integrator import velocity_verlet_step1, velocity_verlet_step2
from md_sim.simulation import Simulation

requires_cuda = pytest.mark.skipif(
    not _CUDA_AVAILABLE,
    reason="CUDA unavailable (set NUMBA_ENABLE_CUDASIM=1 to run on the simulator)",
)


def packed_cluster(n, spacing, margin, seed=4):
    """
    `n` particles on a jittered grid with neighbour spacing `spacing`, centred
    in a periodic box that leaves `margin` empty on every side.

    Both properties matter for a force comparison:

    * `spacing` < interaction range, so the configuration actually interacts.
      `init_particles(n, box)` derives its spacing from the box and can place
      every particle beyond the cut-off, which makes a comparison pass by
      checking 0 against 0.
    * `margin` >= interaction range and box/2 >= interaction range, so the
      minimum-image convention is valid: every pair has exactly one image
      inside the cut-off.  Testing in a box smaller than twice the range
      pins behaviour from a regime where the model itself is wrong.
    """
    rng = np.random.default_rng(seed)
    cols = int(np.ceil(np.sqrt(n)))
    idx = np.arange(n)
    pts = np.column_stack([(idx % cols) * spacing,
                           (idx // cols) * spacing]).astype(np.float64)
    pts += rng.uniform(-0.05 * spacing, 0.05 * spacing, pts.shape)
    box = float((cols - 1) * spacing + 2 * margin)
    pts -= pts.min(axis=0)
    pts += (box - pts.max(axis=0)) / 2.0
    return np.ascontiguousarray(pts, dtype=np.float32), box


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def small_system():
    """4-particle system in a 10×10 box — fast for unit tests."""
    pos, vel = init_particles(4, box_size=10.0, seed=0, target_temp=1.0)
    return pos, vel


@pytest.fixture
def two_particles_touching():
    """Two particles placed just inside the soft-potential range."""
    pos = np.array([[5.0, 5.0], [6.5, 5.0]], dtype=np.float32)  # r = 1.5 < r0 = 2.0
    return pos


# ═════════════════════════════════════════════════════════════════════════════
# utils.py
# ═════════════════════════════════════════════════════════════════════════════

class TestInitParticles:
    def test_output_shapes(self):
        pos, vel = init_particles(9, box_size=10.0, seed=1)
        assert pos.shape == (9, 2)
        assert vel.shape == (9, 2)

    def test_dtype(self):
        pos, vel = init_particles(4, box_size=10.0)
        assert pos.dtype == np.float32
        assert vel.dtype == np.float32

    def test_positions_inside_box(self):
        pos, _ = init_particles(16, box_size=20.0, seed=42)
        assert np.all(pos >= 0.0)
        assert np.all(pos < 20.0)

    def test_zero_net_momentum(self):
        """Center-of-mass velocity should be (near) zero."""
        _, vel = init_particles(16, box_size=20.0, seed=7)
        com = vel.mean(axis=0)
        np.testing.assert_allclose(com, 0.0, atol=1e-5)

    def test_temperature_scaling(self):
        """Temperature of initial velocities should match target_temp."""
        _, vel = init_particles(64, box_size=25.0, seed=0, target_temp=2.0)
        T = temperature(vel)
        assert abs(T - 2.0) < 0.05, f"Expected T≈2.0, got {T:.4f}"

    def test_reproducibility(self):
        pos1, vel1 = init_particles(8, box_size=10.0, seed=123)
        pos2, vel2 = init_particles(8, box_size=10.0, seed=123)
        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(vel1, vel2)

    def test_different_seeds_differ(self):
        pos1, _ = init_particles(8, box_size=10.0, seed=1)
        pos2, _ = init_particles(8, box_size=10.0, seed=2)
        assert not np.allclose(pos1, pos2)


class TestKineticEnergy:
    def test_zero_velocities(self):
        vel = np.zeros((4, 2), dtype=np.float32)
        assert kinetic_energy(vel) == 0.0

    def test_known_value(self):
        # 2 particles each with speed 2 → KE = 0.5 * 1 * (4+4+4+4) = 8
        vel = np.ones((2, 2), dtype=np.float32) * 2.0
        assert kinetic_energy(vel) == pytest.approx(8.0)

    def test_mass_scaling(self):
        vel = np.ones((4, 2), dtype=np.float32)
        ke1 = kinetic_energy(vel, mass=1.0)
        ke2 = kinetic_energy(vel, mass=2.0)
        assert ke2 == pytest.approx(2.0 * ke1)

    def test_positive(self):
        vel = np.random.randn(10, 2).astype(np.float32)
        assert kinetic_energy(vel) > 0.0


class TestTemperature:
    def test_zero_velocities_gives_zero(self):
        vel = np.zeros((4, 2), dtype=np.float32)
        assert temperature(vel) == 0.0

    def test_single_particle_dof(self):
        """N=1, dim=2 → dof = 1*2 - 2 = 0 → T = 0 (no d.o.f.)."""
        vel = np.array([[1.0, 1.0]], dtype=np.float32)
        assert temperature(vel) == 0.0

    def test_equipartition_approx(self):
        """For N particles in 2D, KE ≈ N * T (equipartition)."""
        _, vel = init_particles(100, box_size=30.0, seed=5, target_temp=1.5)
        T = temperature(vel)
        assert abs(T - 1.5) < 0.05

    def test_temperature_positive(self):
        vel = np.random.randn(10, 2).astype(np.float32)
        assert temperature(vel) >= 0.0


# ═════════════════════════════════════════════════════════════════════════════
# cpu_forces.py
# ═════════════════════════════════════════════════════════════════════════════

class TestSoftForces:
    def test_output_shapes(self, two_particles_touching):
        pos = two_particles_touching
        forces, pe = compute_forces_cpu(pos, box_size=10.0, potential_type="soft")
        assert forces.shape == (2, 2)
        assert isinstance(pe, float)

    def test_no_force_beyond_r0(self):
        """Particles farther apart than r0 should feel zero force."""
        pos = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32)  # r = 3 > r0 = 2
        forces, pe = compute_forces_cpu(pos, box_size=20.0, potential_type="soft", r0=2.0)
        np.testing.assert_allclose(forces, 0.0, atol=1e-6)
        assert pe == pytest.approx(0.0)

    def test_newtons_third_law_soft(self, two_particles_touching):
        """Forces on particle i and j must be equal and opposite."""
        forces, _ = compute_forces_cpu(two_particles_touching, box_size=10.0,
                                       potential_type="soft")
        np.testing.assert_allclose(forces[0] + forces[1], 0.0, atol=1e-5)

    def test_repulsive_direction_soft(self, two_particles_touching):
        """Soft force should push particles apart (outward direction)."""
        forces, _ = compute_forces_cpu(two_particles_touching, box_size=10.0,
                                       potential_type="soft")
        # particle 0 is to the left → force on it should point left (negative x)
        assert forces[0, 0] < 0.0
        # particle 1 is to the right → force should point right (positive x)
        assert forces[1, 0] > 0.0

    def test_positive_potential_soft(self, two_particles_touching):
        """Soft repulsion potential must be positive when particles overlap."""
        _, pe = compute_forces_cpu(two_particles_touching, box_size=10.0,
                                   potential_type="soft")
        assert pe > 0.0

    def test_no_self_force(self):
        """A single particle should have zero force."""
        pos = np.array([[5.0, 5.0]], dtype=np.float32)
        forces, pe = compute_forces_cpu(pos, box_size=10.0, potential_type="soft")
        np.testing.assert_allclose(forces, 0.0, atol=1e-6)
        assert pe == pytest.approx(0.0)

    def test_zero_total_force_soft(self):
        """Total force on system must be zero (Newton's 3rd law sum)."""
        pos, _ = init_particles(8, box_size=10.0, seed=3)
        forces, _ = compute_forces_cpu(pos, box_size=10.0, potential_type="soft")
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-4)

    def test_periodic_boundary_soft(self):
        """Force across a periodic boundary should equal the equivalent close-range force."""
        box = 10.0
        # Place two particles ~1 unit apart across the boundary
        pos_direct = np.array([[0.5, 5.0], [1.5, 5.0]], dtype=np.float32)
        pos_wrapped = np.array([[0.5, 5.0], [9.5, 5.0]], dtype=np.float32)  # r=1 via PBC

        f_direct, pe_direct = compute_forces_cpu(pos_direct, box_size=box,
                                                  potential_type="soft")
        f_wrapped, pe_wrapped = compute_forces_cpu(pos_wrapped, box_size=box,
                                                    potential_type="soft")
        # Potential energies should be equal (same actual separation)
        assert pe_direct == pytest.approx(pe_wrapped, rel=1e-4)

    def test_strength_parameter_A(self, two_particles_touching):
        """Larger A should give larger forces and potential."""
        f1, pe1 = compute_forces_cpu(two_particles_touching, box_size=10.0,
                                     potential_type="soft", A=50.0)
        f2, pe2 = compute_forces_cpu(two_particles_touching, box_size=10.0,
                                     potential_type="soft", A=100.0)
        assert pe2 > pe1
        assert np.linalg.norm(f2) > np.linalg.norm(f1)


class TestLJForces:
    def _lj_pair(self, r):
        """Two particles at separation r along x-axis in a big box."""
        pos = np.array([[5.0, 5.0], [5.0 + r, 5.0]], dtype=np.float32)
        return compute_forces_cpu(pos, box_size=50.0, potential_type="lj",
                                  r_cut=5.0, epsilon=1.0, sigma=1.0)

    def test_output_shapes(self):
        pos = np.array([[5.0, 5.0], [6.0, 5.0]], dtype=np.float32)
        forces, pe = compute_forces_cpu(pos, box_size=50.0, potential_type="lj")
        assert forces.shape == (2, 2)
        assert isinstance(pe, float)

    def test_newtons_third_law_lj(self):
        pos = np.array([[5.0, 5.0], [6.0, 5.0]], dtype=np.float32)
        forces, _ = compute_forces_cpu(pos, box_size=50.0, potential_type="lj")
        np.testing.assert_allclose(forces[0] + forces[1], 0.0, atol=1e-4)

    def test_lj_repulsive_at_short_range(self):
        """
        At r < σ the LJ potential is strongly repulsive, so particle 0 (left)
        is pushed left (−x) and particle 1 (right) is pushed right (+x).

        Regression test for an inverted force sign in _lj_forces, which used
        `forces[i] += f_scalar * dx` with dx = pos[j] - pos[i] and therefore
        pulled overlapping particles together instead of pushing them apart.
        """
        forces, pe = self._lj_pair(r=0.9)
        assert forces[0, 0] < 0.0, "left particle must be pushed left"
        assert forces[1, 0] > 0.0, "right particle must be pushed right"
        assert abs(forces[0, 0]) > 1.0, "Repulsive force magnitude should be large at r<sigma"
        assert pe > 0.0, "LJ potential is positive below σ"

    def test_lj_attractive_at_medium_range(self):
        """
        At r ≈ 1.2σ (past the minimum at 2^(1/6)σ ≈ 1.122σ) LJ is weakly
        attractive, so particle 0 is pulled right (+x), toward particle 1.
        """
        forces, pe = self._lj_pair(r=1.2)
        assert forces[0, 0] > 0.0, "left particle must be pulled right"
        assert forces[1, 0] < 0.0, "right particle must be pulled left"
        assert pe < 0.0, "LJ potential is negative past the minimum"

    def test_lj_zero_beyond_rcut(self):
        """Particles outside r_cut should feel no force."""
        forces, pe = self._lj_pair(r=3.5)   # well beyond r_cut=5.0? use default r_cut=2.5
        pos = np.array([[5.0, 5.0], [5.0 + 3.5, 5.0]], dtype=np.float32)
        forces, pe = compute_forces_cpu(pos, box_size=50.0, potential_type="lj",
                                        r_cut=2.5, epsilon=1.0, sigma=1.0)
        np.testing.assert_allclose(forces, 0.0, atol=1e-6)
        assert pe == pytest.approx(0.0)

    def test_zero_total_force_lj(self):
        """Total force on system must be zero."""
        pos, _ = init_particles(6, box_size=20.0, seed=9)
        forces, _ = compute_forces_cpu(pos, box_size=20.0, potential_type="lj")
        np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-4)

    def test_lj_minimum_at_2_16_sigma(self):
        """LJ minimum (F=0) is at r = 2^(1/6) ≈ 1.122."""
        r_min = 2.0 ** (1.0 / 6.0)
        forces_min, _ = self._lj_pair(r=r_min)
        # Force should be close to zero at minimum
        np.testing.assert_allclose(np.abs(forces_min[0, 0]), 0.0, atol=0.5)


class TestForcesMatchPotentialGradient:
    """
    The single strongest invariant in an MD force routine: F must equal
    −∇U of the *same* potential.  These tests recompute U independently and
    differentiate it numerically, so they catch sign errors, missing or
    spurious factors of r, and mismatched prefactors in either potential.
    """

    @staticmethod
    def _potential(pos, box, kind, **kw):
        """
        Reference U(r), written from the documented formulas only.

        Validates `kind` for the same reason compute_forces_cpu does: an
        oracle that silently treats an unrecognised name as LJ would happily
        confirm a wrong-potential implementation.
        """
        if kind not in ("soft", "lj"):
            raise ValueError(f"unknown kind {kind!r}")
        n = len(pos)
        u = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[j] - pos[i]
                d -= box * np.round(d / box)          # minimum image
                r = float(np.hypot(d[0], d[1]))
                if kind == "soft":
                    if r < kw["r0"]:
                        u += kw["A"] * (1.0 - r / kw["r0"]) ** 2
                else:
                    if r <= kw["r_cut"]:
                        sr6 = (kw["sigma"] / r) ** 6
                        u += 4.0 * kw["epsilon"] * (sr6 * sr6 - sr6)
        return u

    def _numerical_forces(self, pos, box, kind, h=1e-6, **kw):
        """F = −dU/dx by central differences."""
        p = pos.astype(np.float64)
        F = np.zeros_like(p)
        for i in range(len(p)):
            for k in range(2):
                orig = p[i, k]
                p[i, k] = orig + h; u_plus  = self._potential(p, box, kind, **kw)
                p[i, k] = orig - h; u_minus = self._potential(p, box, kind, **kw)
                p[i, k] = orig
                F[i, k] = -(u_plus - u_minus) / (2.0 * h)
        return F

    def test_soft_forces_match_gradient(self):
        pos = np.array([[5.0, 5.0], [6.3, 5.4], [4.2, 6.1]], dtype=np.float32)
        kw  = dict(r0=2.0, A=50.0)
        f_code, pe = compute_forces_cpu(pos, 20.0, potential_type="soft", **kw)
        f_num = self._numerical_forces(pos, 20.0, "soft", **kw)
        np.testing.assert_allclose(f_code, f_num, rtol=1e-3, atol=1e-3)
        assert pe == pytest.approx(
            self._potential(pos.astype(np.float64), 20.0, "soft", **kw), rel=1e-5
        )

    @pytest.mark.parametrize("r", [0.9, 1.0, 2.0 ** (1.0 / 6.0), 1.4, 2.0])
    def test_lj_forces_match_gradient(self, r):
        """Covers the repulsive branch, the minimum, and the attractive tail."""
        pos = np.array([[5.0, 5.0], [5.0 + r, 5.0]], dtype=np.float32)
        kw  = dict(r_cut=2.5, epsilon=1.0, sigma=1.0)
        f_code, _ = compute_forces_cpu(pos, 50.0, potential_type="lj", **kw)
        f_num = self._numerical_forces(pos, 50.0, "lj", **kw)
        np.testing.assert_allclose(f_code, f_num, rtol=1e-3,
                                   atol=1e-3 * max(1.0, np.abs(f_num).max()))

    def test_lj_forces_match_gradient_many_body(self):
        """
        Nine interacting particles, minimum image valid (box/2 > r_cut).

        The tolerance is scaled to the forces actually present and has no
        `max(1.0, ...)` floor: with init_particles(6, box_size=8.0) only 4 of
        15 pairs fell inside r_cut and all sat in the far tail at |F| ~ 0.06,
        so a 1e-2 floor meant an effective 16% tolerance and a 15% error in
        the LJ prefactor passed unnoticed.  Measured agreement here is ~4e-8
        relative, so 1e-4 still leaves a ~2000x margin.
        """
        kw = dict(r_cut=2.5, epsilon=1.0, sigma=1.0)
        pos, box = packed_cluster(9, spacing=1.1, margin=kw["r_cut"] + 0.5)
        f_code, _ = compute_forces_cpu(pos, box, potential_type="lj", **kw)
        f_num = self._numerical_forces(pos, box, "lj", **kw)
        assert np.abs(f_num).max() > 1.0, "configuration has no real interactions"
        np.testing.assert_allclose(f_code, f_num, rtol=1e-4,
                                   atol=1e-4 * np.abs(f_num).max())


class TestPotentialTypeValidation:
    """An unrecognised potential_type used to fall through to LJ silently."""

    def test_unknown_potential_type_rejected(self):
        pos = np.array([[5.0, 5.0], [6.0, 5.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="unknown potential_type"):
            compute_forces_cpu(pos, box_size=10.0, potential_type="harmonic")

    def test_case_mismatch_rejected(self):
        """'LJ' is not 'lj': it used to silently select the LJ branch while
        Simulation still reported (and GPU-dispatched) something else."""
        pos = np.array([[5.0, 5.0], [6.0, 5.0]], dtype=np.float32)
        with pytest.raises(ValueError):
            compute_forces_cpu(pos, box_size=10.0, potential_type="LJ")

    def test_simulation_rejects_unknown_potential(self):
        with pytest.raises(ValueError, match="unknown potential_type"):
            Simulation(n_particles=4, box_size=10.0, potential_type="Soft")


# ═════════════════════════════════════════════════════════════════════════════
# integrator.py
# ═════════════════════════════════════════════════════════════════════════════

class TestVelocityVerlet:
    def _make_arrays(self, n=2):
        pos = np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)[:n]
        vel = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=np.float32)[:n]
        forces = np.zeros((n, 2), dtype=np.float32)
        return pos, vel, forces

    def test_step1_updates_positions(self):
        pos, vel, forces = self._make_arrays()
        pos0 = pos.copy()
        velocity_verlet_step1(pos, vel, forces, dt=0.01, box_size=10.0)
        assert not np.allclose(pos, pos0)

    def test_step1_zero_force_free_particle(self):
        """With zero force, x(t+dt) = x(t) + v*dt."""
        pos = np.array([[1.0, 0.0]], dtype=np.float32)
        vel = np.array([[2.0, 0.0]], dtype=np.float32)
        forces = np.zeros((1, 2), dtype=np.float32)
        dt = 0.1
        velocity_verlet_step1(pos, vel, forces, dt=dt, box_size=100.0)
        assert pos[0, 0] == pytest.approx(1.0 + 2.0 * dt, rel=1e-5)

    def test_step2_updates_velocities(self):
        pos, vel, forces = self._make_arrays()
        vel0 = vel.copy()
        velocity_verlet_step1(pos, vel, forces, dt=0.01, box_size=10.0)
        forces_new = np.ones_like(forces)  # arbitrary new forces
        velocity_verlet_step2(vel, forces_new, dt=0.01)
        # velocities should have changed due to new forces
        assert not np.allclose(vel, vel0)

    def test_periodic_wrap(self):
        """Positions that exceed box_size should wrap."""
        pos = np.array([[9.9, 5.0]], dtype=np.float32)
        vel = np.array([[5.0, 0.0]], dtype=np.float32)
        forces = np.zeros((1, 2), dtype=np.float32)
        velocity_verlet_step1(pos, vel, forces, dt=0.1, box_size=10.0)
        assert 0.0 <= pos[0, 0] < 10.0

    def test_acceleration_applied(self):
        """Force should accelerate the particle."""
        pos = np.array([[5.0, 5.0]], dtype=np.float32)
        vel = np.array([[0.0, 0.0]], dtype=np.float32)
        forces = np.array([[10.0, 0.0]], dtype=np.float32)
        dt = 0.1
        velocity_verlet_step1(pos, vel, forces, dt=dt, box_size=100.0)
        # x(dt) = 0 + 0*dt + 0.5*a*dt^2 = 0.5*10*0.01 = 0.05
        assert pos[0, 0] == pytest.approx(5.0 + 0.5 * 10.0 * dt**2, rel=1e-4)

    def test_full_step_energy_conservation(self):
        """
        For a soft-repulsion system, total energy should be roughly conserved
        over a few steps (short enough that drift is negligible).
        """
        n = 4
        pos, vel = init_particles(n, box_size=10.0, seed=0, target_temp=0.5)
        forces, pe = compute_forces_cpu(pos, box_size=10.0, potential_type="soft")
        ke_init = kinetic_energy(vel)
        te_init = ke_init + pe

        dt = 0.001
        for _ in range(10):
            velocity_verlet_step1(pos, vel, forces, dt=dt, box_size=10.0)
            forces, pe = compute_forces_cpu(pos, box_size=10.0, potential_type="soft")
            velocity_verlet_step2(vel, forces, dt=dt)

        ke_final = kinetic_energy(vel)
        te_final = ke_final + pe
        # Allow 2% drift over 10 steps
        assert abs(te_final - te_init) / (abs(te_init) + 1e-8) < 0.02


# ═════════════════════════════════════════════════════════════════════════════
# simulation.py
# ═════════════════════════════════════════════════════════════════════════════

class TestSimulationInit:
    def test_default_construction(self, capsys):
        sim = Simulation(n_particles=16, box_size=10.0, seed=0)
        out = capsys.readouterr().out
        assert "16 particles" in out
        assert "CPU" in out

    def test_positions_shape(self):
        sim = Simulation(n_particles=9, box_size=10.0, seed=0)
        assert sim.positions.shape == (9, 2)

    def test_velocities_shape(self):
        sim = Simulation(n_particles=9, box_size=10.0, seed=0)
        assert sim.velocities.shape == (9, 2)

    def test_forces_computed_at_init(self):
        sim = Simulation(n_particles=4, box_size=10.0, seed=0)
        assert sim.forces.shape == (4, 2)

    def test_lj_mode_accepted(self):
        sim = Simulation(n_particles=4, box_size=10.0, potential_type="lj",
                         dt=0.001, seed=0)
        assert sim.potential_type == "lj"

    def test_gpu_flag_tracks_cuda_availability(self, capsys):
        """
        use_gpu=True must be honoured where CUDA exists and fall back to the
        CPU (with a warning) where it does not.

        This previously asserted `sim.use_gpu is False` with the comment
        "no GPU in CI", which made the test fail on exactly the CUDA hardware
        the project targets.
        """
        sim = Simulation(n_particles=4, box_size=10.0, use_gpu=True, seed=0)
        out = capsys.readouterr().out
        assert sim.use_gpu is _CUDA_AVAILABLE
        if not _CUDA_AVAILABLE:
            assert "CUDA not available" in out

    def test_gpu_declined_for_lj(self, capsys):
        """The CUDA kernel only implements 'soft', so LJ must stay on the CPU
        regardless of whether a GPU is present."""
        sim = Simulation(n_particles=4, box_size=10.0, dt=0.001,
                         potential_type="lj", use_gpu=True, seed=0)
        assert sim.use_gpu is False
        assert "only supports 'soft'" in capsys.readouterr().out


class TestSimulationStep:
    def test_step_increments_counter(self):
        sim = Simulation(n_particles=4, box_size=10.0, seed=0)
        sim.step()
        assert sim._step == 1

    def test_step_advances_time(self):
        dt = 0.01
        sim = Simulation(n_particles=4, box_size=10.0, dt=dt, seed=0)
        sim.step()
        assert sim._time == pytest.approx(dt)

    def test_step_returns_state(self):
        sim = Simulation(n_particles=4, box_size=10.0, seed=0)
        state = sim.step()
        assert state.step == 1
        assert state.ke > 0.0
        assert state.temp > 0.0

    def test_thermostat_rescales_temp(self):
        """With thermostat ON, temperature should stay close to target."""
        sim = Simulation(n_particles=16, box_size=10.0, target_temp=1.0, seed=0)
        for _ in range(20):
            sim.step(thermostat=True)
        T = temperature(sim.velocities)
        assert abs(T - 1.0) < 0.05


class TestSimulationEquilibrate:
    def test_equilibrate_runs(self):
        sim = Simulation(n_particles=4, box_size=10.0, seed=0)
        sim.equilibrate(steps=5, verbose=False)
        assert sim._step == 5

    def test_equilibrate_keeps_temp_close(self):
        sim = Simulation(n_particles=32, box_size=15.0, target_temp=1.0, seed=0)
        sim.equilibrate(steps=50, verbose=False)
        T = temperature(sim.velocities)
        assert abs(T - 1.0) < 0.15

    def test_equilibrate_verbose(self, capsys):
        sim = Simulation(n_particles=4, box_size=10.0, seed=0)
        sim.equilibrate(steps=3, verbose=True)
        out = capsys.readouterr().out
        assert "Equilibrating" in out
        assert "Done" in out


class TestSimulationRun:
    def test_run_returns_history(self):
        """
        run() logs at s==0 (step 1) AND every log_every steps.
        With steps=10, log_every=5: logged at step 1, 5, 10 → 3 entries.
        """
        sim = Simulation(n_particles=4, box_size=10.0, seed=0)
        history = sim.run(steps=10, log_every=5, verbose=False)
        assert len(history) == 3  # step 1, step 5, step 10

    def test_history_step_numbers(self):
        sim = Simulation(n_particles=4, box_size=10.0, seed=0)
        history = sim.run(steps=20, log_every=10, verbose=False)
        steps = [s.step for s in history]
        assert 1 in steps or 10 in steps  # first and every log_every

    def test_run_clears_history_between_runs(self):
        """history.clear() is called at the start of each run(), so the second
        run's history only contains entries from that run (not cumulative)."""
        sim = Simulation(n_particles=4, box_size=10.0, seed=0)
        sim.run(steps=10, log_every=5, verbose=False)
        history2 = sim.run(steps=10, log_every=5, verbose=False)
        # Second run: step 11 (s==0), step 15, step 20 → 3 entries
        assert len(history2) == 3
        # And step numbers come from the second run only
        assert history2[0].step == 11

    def test_energy_fields_present(self):
        sim = Simulation(n_particles=4, box_size=10.0, seed=0)
        history = sim.run(steps=5, log_every=5, verbose=False)
        s = history[-1]
        assert s.ke >= 0.0
        assert s.temp >= 0.0
        assert isinstance(s.pe, float)

    def test_run_verbose_output(self, capsys):
        sim = Simulation(n_particles=4, box_size=10.0, seed=0)
        sim.run(steps=10, log_every=10, verbose=True)
        out = capsys.readouterr().out
        assert "Step" in out
        assert "Done" in out

    def test_energy_approx_conserved_nve(self):
        """Total energy drift over short NVE run should be small."""
        sim = Simulation(n_particles=8, box_size=12.0, dt=0.005,
                         target_temp=0.5, seed=0)
        sim.equilibrate(steps=20, verbose=False)
        history = sim.run(steps=50, log_every=50, verbose=False)
        total_energies = [s.ke + s.pe for s in history]
        if len(total_energies) >= 2:
            drift = abs(total_energies[-1] - total_energies[0])
            relative = drift / (abs(total_energies[0]) + 1e-8)
            assert relative < 0.05, f"Energy drift too large: {relative:.3%}"


# ═════════════════════════════════════════════════════════════════════════════
# Edge-case / regression tests
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_two_particle_simulation(self):
        """Minimum-size simulation should not crash."""
        sim = Simulation(n_particles=2, box_size=10.0, seed=0)
        sim.run(steps=10, log_every=5, verbose=False)

    def test_large_particle_count(self):
        """Larger system should initialise and take one step without error."""
        sim = Simulation(n_particles=256, box_size=30.0, seed=0)
        state = sim.step()
        assert state.ke > 0.0

    def test_small_timestep_stability(self):
        """Very small dt should keep energy well conserved."""
        sim = Simulation(n_particles=4, box_size=10.0, dt=0.0001, seed=0)
        sim.run(steps=100, log_every=100, verbose=False)
        # Just check it didn't crash or produce NaNs
        assert not np.any(np.isnan(sim.positions))
        assert not np.any(np.isnan(sim.velocities))

    def test_no_nan_after_run(self):
        sim = Simulation(n_particles=16, box_size=15.0, seed=0)
        sim.equilibrate(steps=10, verbose=False)
        sim.run(steps=20, log_every=20, verbose=False)
        assert not np.any(np.isnan(sim.positions))
        assert not np.any(np.isnan(sim.velocities))

    def test_lj_simulation_runs(self):
        """LJ potential end-to-end run should complete without error."""
        sim = Simulation(n_particles=4, box_size=10.0, dt=0.001,
                         potential_type="lj", seed=0)
        sim.equilibrate(steps=10, verbose=False)
        history = sim.run(steps=10, log_every=10, verbose=False)
        assert len(history) >= 1


# ═════════════════════════════════════════════════════════════════════════════
# Energy conservation — the integrator only conserves E when F = −∇U,
# so these runs fail loudly if a force routine drifts out of sync again.
# ═════════════════════════════════════════════════════════════════════════════

class TestEnergyConservation:
    def _drift(self, sim, steps):
        """
        Relative |E(steps) - E(0)| over a thermostat-free run.

        The baseline is read *before* the first step: taking it after step 1
        (as this used to) silently excluded the first step's error from every
        measurement.  PE is NaN on the GPU path (Simulation._compute_forces
        returns float("nan")), which would turn every assertion below into a
        silent `NaN < tol` failure, so reject that explicitly.
        """
        assert steps >= 1, "need at least one step to measure drift"
        start = sim._log()
        e0 = start.ke + start.pe
        assert not np.isnan(e0), (
            "potential energy is NaN - the GPU path does not compute PE, so "
            "energy drift cannot be measured for a use_gpu=True simulation"
        )
        for _ in range(steps):
            state = sim.step(thermostat=False)
        return abs((state.ke + state.pe) - e0) / (abs(e0) + 1e-12)

    def test_soft_nve_conserves_energy(self):
        """
        The soft potential is continuous and finite everywhere, so
        Velocity-Verlet conserves E very tightly (measured ~1e-6 relative).
        A loose bound here would hide a mis-scaled force, so keep it tight:
        measured drift for this configuration is ~6e-7, and injecting the old
        1/r kernel mis-scaling into the CPU path gives ~4e-2.
        """
        sim = Simulation(n_particles=16, box_size=12.0, dt=0.002,
                         target_temp=0.5, seed=0)
        assert self._drift(sim, 300) < 1e-5

    def test_lj_bound_pair_stays_bound(self):
        """
        Regression test for the inverted LJ force sign, built to be immune to
        the cut-off artefact below.

        Two particles released at rest at r = 1.2σ sit just outside the LJ
        minimum (2^(1/6)σ ≈ 1.122σ) with total energy U(1.2σ) < 0, so the
        correct dynamics is a bound oscillation that can never exceed its
        release separation.  With the sign inverted the pair is pushed apart
        and runs away instead.

        The run is 3 time units (dt=1e-3 x 3000).  An earlier version used
        dt=1e-4 x 2000 = 0.2 time units, which was far too short for the
        runaway to show: the buggy pair only reached 1.2916σ, so the
        separation assertions passed and *only* the energy check caught the
        bug.  Measured here: correct stays in [1.070, 1.200] with 3e-5 drift,
        inverted reaches 6.37σ with 198% drift.
        """
        sim = Simulation(n_particles=2, box_size=30.0, dt=1e-3,
                         potential_type="lj", target_temp=1.0, seed=0)
        sim.positions = np.array([[15.0, 15.0], [16.2, 15.0]], dtype=np.float32)
        sim.velocities = np.zeros((2, 2), dtype=np.float32)
        sim.forces, sim._pe = sim._compute_forces()

        seps, energies = [], []
        for _ in range(3000):
            state = sim.step(thermostat=False)
            seps.append(abs(float(sim.positions[1, 0] - sim.positions[0, 0])))
            energies.append(state.ke + state.pe)

        # A bound pair released from rest cannot exceed its release separation.
        assert max(seps) < 1.25, (
            f"pair ran away: max separation {max(seps):.3f}σ (expected < 1.25σ)"
        )
        assert min(seps) > 1.0, f"pair collapsed: min separation {min(seps):.3f}σ"
        # No pair ever crosses r_cut here, so energy must be well conserved.
        drift = abs(energies[-1] - energies[0]) / abs(energies[0])
        assert drift < 1e-3, f"energy drift {drift:.3%} in a bound LJ pair"

    def test_lj_truncation_drift_is_bounded(self):
        """
        Documents a *modelling* limitation, not a bug: this LJ is truncated at
        r_cut but not shifted, so U jumps by U(r_cut) whenever a pair crosses
        the cut-off and NVE energy drifts slightly (~1% over 1200 steps).
        Shifting the potential by U(r_cut) would remove it.

        Measured for this exact configuration: 0.11% at 300 steps, 0.87% at
        1000, 1.09% at 1200.  The assertion runs the 1200 steps it quotes and
        bounds them at 2%; a previous version measured at 1200 but asserted
        over 300 steps, leaving the bound ~45x looser than the phenomenon.
        """
        sim = Simulation(n_particles=16, box_size=12.0, dt=0.001,
                         potential_type="lj", target_temp=1.0, seed=0)
        assert self._drift(sim, 1200) < 0.02

    def test_lj_overlapping_pair_separates(self):
        """Two particles well inside σ must fly apart, not collapse."""
        sim = Simulation(n_particles=2, box_size=20.0, dt=1e-5,
                         potential_type="lj", target_temp=0.0, seed=0)
        sim.positions = np.array([[9.6, 10.0], [10.4, 10.0]], dtype=np.float32)
        sim.velocities = np.zeros((2, 2), dtype=np.float32)
        sim.forces, sim._pe = sim._compute_forces()

        def sep(s):
            d = s.positions[1] - s.positions[0]
            d -= s.box_size * np.round(d / s.box_size)
            return float(np.hypot(d[0], d[1]))

        r_before = sep(sim)
        for _ in range(200):
            sim.step(thermostat=False)
        assert sep(sim) > r_before, "overlapping LJ particles must repel"


# ═════════════════════════════════════════════════════════════════════════════
# GPU kernel — skipped unless CUDA (or the CUDA simulator) is available
# ═════════════════════════════════════════════════════════════════════════════

@requires_cuda
class TestGPUMatchesCPU:
    """
    The GPU kernel must reproduce the CPU 'soft' path, which is itself pinned
    to −∇U by TestForcesMatchPotentialGradient.  The kernel previously divided
    by r0*r2 instead of r0*r, scaling every pair force by a spurious 1/r —
    invisible to any test that only checked shapes, symmetry or finiteness.
    """

    R0, A = 2.0, 50.0

    def _config(self, n):
        """Interacting cluster with a valid minimum image (see packed_cluster)."""
        return packed_cluster(n, spacing=1.3, margin=self.R0 + 0.5)

    # n = 33 and n = 130 also cross the 128-thread/block boundary, so they
    # exercise the multi-block launch and the `i >= n` guard.
    @pytest.mark.parametrize("n", [2, 4, 33, 130])
    def test_matches_cpu(self, n):
        pos, box = self._config(n)
        f_cpu, _ = compute_forces_cpu(pos, box, potential_type="soft",
                                      r0=self.R0, A=self.A)
        f_gpu, pe = compute_forces_gpu(pos, box, r0=self.R0, A=self.A)
        assert pe is None
        assert f_gpu.shape == (n, 2)
        assert f_gpu.dtype == np.float32
        # Guard against a vacuous pass: if every particle sat beyond r0 the
        # comparison would be 0 == 0 and the 1/r scaling bug would slip by.
        scale = float(np.abs(f_cpu).max())
        assert scale > 1.0, "test configuration has no interactions"
        # Measured float32 agreement is ~3e-7 relative; 1e-4 leaves a ~300x
        # margin.  No max(1.0, ...) floor - that would let the tolerance rise
        # above the signal whenever the forces are small.
        np.testing.assert_allclose(f_gpu, f_cpu, rtol=1e-4, atol=1e-4 * scale)

    def test_newtons_third_law(self):
        pos, box = self._config(32)
        f_gpu, _ = compute_forces_gpu(pos, box, r0=self.R0, A=self.A)
        scale = float(np.abs(f_gpu).max())
        assert scale > 1.0, "test configuration has no interactions"
        np.testing.assert_allclose(f_gpu.sum(axis=0), 0.0, atol=1e-4 * scale)

    def test_accepts_float64_and_noncontiguous_input(self):
        """
        Documents which input forms compute_forces_gpu accepts, and why the
        wrapper's np.ascontiguousarray(..., dtype=float32) is there.

        Contiguity is the load-bearing half.  numba's sentry_contiguous
        rejects an array that is neither C- nor F-contiguous ("Array contains
        non-contiguous buffer and cannot be transferred as a single memory
        region"), so without the coercion such an input raises instead of
        computing.  Note that np.asfortranarray(pos) and pos.T.copy().T are
        both F-contiguous and *would* be accepted, so neither of those
        exercises this path - a column slice of a wider array does.

        Two honest caveats, so this test is not mistaken for more than it is:
          * The CUDA simulator does not enforce contiguity, so under
            NUMBA_ENABLE_CUDASIM=1 nothing here fails if the coercion is
            deleted; it is load-bearing only on real hardware.
          * The float32 cast is not covered by comparing a float64 run against
            a float32 run (@cuda.jit just compiles a second specialisation
            that agrees to well within tolerance) nor by the output dtype
            (forces_out is an explicitly float32 device array either way).
            Dropping the cast costs a duplicate kernel compilation, not
            correctness.
        """
        pos, box = self._config(8)
        f32, _ = compute_forces_gpu(pos, box, r0=self.R0, A=self.A)
        scale = float(np.abs(f32).max())
        assert scale > 1.0, "test configuration has no interactions"

        f64, _ = compute_forces_gpu(pos.astype(np.float64), box,
                                    r0=self.R0, A=self.A)
        assert f64.dtype == np.float32
        np.testing.assert_allclose(f64, f32, rtol=1e-4, atol=1e-4 * scale)

        # Neither C- nor F-contiguous: exactly what numba refuses to transfer.
        buf = np.zeros((len(pos), 3), dtype=np.float32)
        buf[:, :2] = pos
        pos_nc = buf[:, :2]
        assert not pos_nc.flags["C_CONTIGUOUS"]
        assert not pos_nc.flags["F_CONTIGUOUS"]
        f_nc, _ = compute_forces_gpu(pos_nc, box, r0=self.R0, A=self.A)
        np.testing.assert_allclose(f_nc, f32, rtol=1e-4, atol=1e-4 * scale)
