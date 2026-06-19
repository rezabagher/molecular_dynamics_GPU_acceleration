# Molecular Dynamics GPU Accelerator

A 2D molecular dynamics simulator built to explore GPU acceleration of scientific workloads using NVIDIA CUDA and Numba. The project implements pairwise force models and Velocity-Verlet integration, and benchmarks a serial CPU implementation against a CUDA-accelerated GPU version.

## Highlights

- CUDA acceleration via Numba `@cuda.jit` kernels
- Soft-repulsion and Lennard-Jones interparticle potentials
- NVT thermostat control
- Velocity-Verlet time integration
- Quantified CPU vs. GPU performance benchmarking on real hardware
- Modular Python codebase designed for experimentation and extension

## Benchmark Results

Benchmarks were performed on an **NVIDIA Tesla T4 GPU** (Google Colab). Timings include kernel execution and host-to-device memory transfers.

| Particles | Speedup (GPU vs. CPU) |
|-----------|----------------------|
| 256       | **66×**              |
| 1,024     | **568×**             |

These results demonstrate how scientific workloads dominated by pairwise interactions — an O(N²) problem — scale well under GPU parallelism.

## Motivation

Pairwise force evaluation is the computational bottleneck in molecular dynamics. For N particles, every particle interacts with every other, producing an O(N²) workload that grows rapidly with system size.

This project investigates how GPU parallelism can accelerate that bottleneck by assigning one CUDA thread per particle and computing all forces concurrently, then comparing execution times against a CPU reference implementation.

The goal is not to compete with production MD packages such as GROMACS, NAMD, or LAMMPS, but to demonstrate the principles of GPU acceleration, numerical simulation, and quantitative performance analysis.

---

## Physics Model

### Soft-Repulsion Potential

Particles interact through a finite-range harmonic repulsion:

$$U(r) = A\left(1 - \frac{r}{r_0}\right)^2, \quad r < r_0$$

$$U(r) = 0, \quad r \geq r_0$$

This potential is numerically stable at relatively large timesteps and avoids the singularities present in harder potentials, making it well-suited for demonstrating GPU acceleration.

### Lennard-Jones Potential

The simulator optionally supports the Lennard-Jones interaction:

$$U(r) = 4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6\right]$$

which models short-range repulsion and long-range attraction commonly used in molecular simulation.

### Time Integration

Particle trajectories are advanced using the Velocity-Verlet scheme:

$$x(t + \Delta t) = x(t) + v(t)\Delta t + \tfrac{1}{2}a(t)\Delta t^2$$

$$v(t + \Delta t) = v(t) + \tfrac{1}{2}\left[a(t) + a(t + \Delta t)\right]\Delta t$$

Velocity-Verlet is time-reversible, symplectic, and the standard integrator in molecular dynamics.

---

## GPU Implementation

The GPU kernel assigns one CUDA thread to each particle. Each thread independently accumulates the net force from all other particles, enabling thousands of force calculations to run concurrently:

```python
@cuda.jit
def _soft_forces_kernel(...):
    i = cuda.grid(1)
    if i < n_particles:
        for j in range(n_particles):
            # compute pairwise force contribution
            ...
```

This maps naturally onto the O(N²) structure of the problem and is the source of the large speedups observed at higher particle counts.

---

## Project Structure

```
md_sim/
├── cpu_forces.py       # Serial CPU force evaluation
├── gpu_forces.py       # CUDA kernel force evaluation
├── integrator.py       # Velocity-Verlet integrator
├── simulation.py       # Simulation loop and NVT thermostat
├── benchmark.py        # CPU vs. GPU timing harness
└── utils.py            # Shared utilities

run_example.py
README.md
```

---

## Running the Project

### Installation

```bash
pip install numpy numba
```

GPU execution requires the NVIDIA CUDA Toolkit and a CUDA-capable GPU.

### CPU Simulation

```bash
python run_example.py
```

### GPU Simulation

```bash
python run_example.py --gpu
```

### Benchmarking

```bash
python run_example.py --benchmark
```

---

## Limitations

This project is an educational and performance-exploration tool, not a production MD engine. Current limitations include:

- O(N²) all-pairs force evaluation (no neighbor lists or cell lists)
- Two-dimensional simulation domain
- Single-GPU execution only
- No distributed-memory parallelism

---

## Potential Extensions

- Verlet neighbor lists and cell-based spatial partitioning
- CUDA shared-memory tiling to reduce global memory traffic
- Multi-GPU execution
- Three-dimensional simulation domain
- Nsight profiling and kernel-level optimization
- Comparison against vectorized NumPy and compiled CPU baselines (e.g. Cython, Numba CPU)

---

## Technologies

- Python · NumPy · Numba CUDA
- NVIDIA CUDA · NVIDIA Tesla T4
- Google Colab
