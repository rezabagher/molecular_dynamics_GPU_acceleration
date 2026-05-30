# Molecular Dynamics GPU Acceleration

This project presents a minimal yet technically rigorous Molecular Dynamics (MD) simulator that reflects my background in **theoretical chemistry** and **computer science**. It is designed to demonstrate how classical MD algorithms can be mapped onto modern GPU architectures using **Numba CUDA**, with a particular focus on accelerating force evaluation—the dominant computational cost in many MD simulations.

---

## Scientific Motivation

Molecular Dynamics is a deterministic simulation method in which the time evolution of a system of particles is obtained by integrating Newton’s equations of motion:

`
m_i * d²r_i/dt² = F_i
`

The forces are derived from an interatomic potential. In this project, I use a simplified pairwise interaction model (Lennard-Jones-like or soft-repulsive) that captures the essential physics of short-range repulsion and intermediate-range attraction. Although simplified, the model still requires evaluating all pairwise interactions, resulting in an `(O(N^2))` computational complexity.

Time integration is performed using the **Velocity Verlet algorithm**, a standard method in computational chemistry and molecular simulation due to its symplectic character, numerical stability, and favorable energy conservation properties. Even as a compact implementation, the structure of this simulator mirrors that of larger MD codes used in theoretical chemistry, statistical mechanics, and materials science.

---

## Computational Motivation

Force evaluation in MD is highly parallelizable: the total force acting on each particle can be computed independently by summing contributions from all other particles. This makes MD particularly well suited for GPU acceleration.

This project uses **Numba CUDA** to implement:

- a GPU kernel in which each thread computes the force on a single particle,
- grid-stride loops to support arbitrary system sizes,
- explicit synchronization for reliable benchmarking.

A NumPy-based CPU implementation is included as a baseline for comparison. The GPU implementation illustrates how even relatively simple CUDA kernels can provide substantial reductions in wall-clock time for medium-sized systems.

---

## Features

- **2D Molecular Dynamics simulation**
- **Pairwise interaction potential** (Lennard-Jones-like or soft-repulsive)
- **Velocity Verlet integration**
- **CPU implementation** with NumPy
- **GPU implementation** with Numba CUDA
- **CPU vs GPU benchmarking**
- **Modular and extensible codebase**
- **Scientifically grounded algorithmic design**

---

## Project Structure
```text
molecular_dynamics_GPU_acceleration/
│
├── cpu_forces.py        # CPU force calculation (NumPy)
├── gpu_forces.py        # GPU force calculation (Numba CUDA)
├── integrator.py        # Velocity Verlet integration
├── simulation.py        # Main simulation loop
├── benchmark.py         # CPU vs GPU performance comparison
├── utils.py             # Initialization, constants, helper functions
└── README.md            # Documentation
