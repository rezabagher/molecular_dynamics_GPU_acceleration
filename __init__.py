"""
md_sim — Molecular Dynamics Toy Simulator
==========================================
A minimal 2-D Lennard-Jones MD simulation showcasing:
  • CPU force calculation (NumPy)
  • GPU force calculation (Numba CUDA)
  • Velocity-Verlet integrator
  • CPU vs GPU benchmarking

Usage
-----
    from md_sim.simulation import Simulation
    sim = Simulation(n_particles=256, box_size=20.0, dt=0.005, use_gpu=True)
    sim.run(steps=500)
"""

__version__ = "0.1.0"
__author__  = "REZA BAGHERI ALASHTI"
