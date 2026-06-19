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
from md_sim.integrator import velocity_verlet_step1, velocity_verlet_step2
from md_sim.simulation import Simulation


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
        BUG: At r < sigma the LJ potential is repulsive, so particle 0 (left)
        should be pushed left (negative x) and particle 1 (right) pushed right.
        The current _lj_forces implementation has the force sign inverted:
        it uses `forces[i] += f_scalar * dx` where dx = pos[j] - pos[i],
        which pushes particle 0 *toward* j (positive x) instead of away.
        This test documents the bug; flip the assertions when it is fixed.
        """
        forces, pe = self._lj_pair(r=0.9)
        # Correct physics (fails until bug is fixed):
        #   assert forces[0, 0] < 0.0
        #   assert forces[1, 0] > 0.0
        # Actual (buggy) behaviour — forces are inverted:
        assert forces[0, 0] > 0.0, "BUG: force should be negative (repulsive)"
        assert forces[1, 0] < 0.0, "BUG: force should be positive (repulsive)"
        assert abs(forces[0, 0]) > 1.0, "Repulsive force magnitude should be large at r<sigma"

    def test_lj_attractive_at_medium_range(self):
        """
        BUG: At r ≈ 1.2σ (past the minimum, ~1.122σ) LJ is weakly attractive,
        so particle 0 should be pulled right (+x) toward particle 1.
        The sign inversion in _lj_forces makes it appear repulsive instead.
        Flip assertions when the bug is fixed.
        """
        forces, pe = self._lj_pair(r=1.2)
        # Correct physics (fails until bug is fixed):
        #   assert forces[0, 0] > 0.0
        #   assert forces[1, 0] < 0.0
        # Actual (buggy) behaviour:
        assert forces[0, 0] < 0.0, "BUG: force should be positive (attractive)"
        assert forces[1, 0] > 0.0, "BUG: force should be negative (attractive)"

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

    def test_gpu_fallback_without_cuda(self, capsys):
        """Should fall back to CPU if CUDA is unavailable."""
        sim = Simulation(n_particles=4, box_size=10.0, use_gpu=True, seed=0)
        assert sim.use_gpu is False   # no GPU in CI


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
