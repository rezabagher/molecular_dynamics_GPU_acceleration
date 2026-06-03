# 🔬 MD-Sim — Molecular Dynamics Toy Simulator

A clean, minimal **2-D particle simulator** written in Python, showcasing GPU
acceleration with **Numba CUDA** alongside a CPU reference implementation.

Built as a portfolio project demonstrating:

- Scientific computing fundamentals (numerical integration, force fields)
- GPU programming with Numba CUDA (`@cuda.jit`, `cuda.grid`, `cuda.synchronize`)
- Velocity-Verlet integration (symplectic, second-order accurate)
- CPU vs GPU benchmarking methodology
- Modular, well-commented Python project structure

---

## ⚙️ Physics Background

### Soft repulsion potential (default)

Each particle pair closer than diameter `r₀` feels a harmonic repulsion:

```
U(r) = A · (1 − r/r₀)²    for r < r₀,  else 0
```

This potential is always finite, never diverges, and stays numerically stable
with any reasonable timestep. It models a colloidal suspension or soft-sphere gas.

### Lennard-Jones potential (optional)

The classic MD pair potential:

```
U(r) = 4ε [ (σ/r)¹² − (σ/r)⁶ ]
```

The `r⁻¹²` term is short-range Pauli repulsion; `r⁻⁶` is van der Waals attraction.
Requires a small timestep (dt ≤ 0.001) and runs in NVT (thermostat always on).

### Integrator

Both potentials use **Velocity-Verlet** integration:

```
x(t+Δt)  =  x(t) + v(t)·Δt + ½a(t)·Δt²
v(t+Δt)  =  v(t) + ½[a(t) + a(t+Δt)]·Δt
```

Time-reversible and second-order accurate — the standard choice for MD.

---

## 📁 Project Structure

```
md_sim/
├── __init__.py       # package metadata
├── cpu_forces.py     # CPU reference: soft + LJ potentials (NumPy)
├── gpu_forces.py     # GPU kernel:   soft potential (Numba CUDA)
├── integrator.py     # Velocity-Verlet, force-agnostic
├── simulation.py     # Main driver: init → equilibrate → run
├── benchmark.py      # CPU vs GPU timing comparison
└── utils.py          # Particle init, kinetic energy, temperature

run_example.py        # Minimal working example (run this!)
README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install numpy numba
```

For GPU support, also install the NVIDIA CUDA Toolkit:
[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

> **No GPU?** The simulator runs fully in CPU-only mode (the default).

### 2. Run a simulation

```bash
# Soft repulsion, 256 particles, CPU
python run_example.py

# Custom particle count, 1000 steps
python run_example.py --n 512 --steps 1000

# GPU acceleration (requires CUDA)
python run_example.py --gpu --n 1024

# Lennard-Jones potential (CPU, NVT)
python run_example.py --lj --n 64 --steps 500

# CPU vs GPU benchmark table
python run_example.py --benchmark
```

### 3. Use as a library

```python
from md_sim.simulation import Simulation

# Soft repulsion (stable, default)
sim = Simulation(
    n_particles    = 256,
    box_size       = 25.0,
    dt             = 0.01,
    use_gpu        = False,     # set True if CUDA is available
    potential_type = "soft",
    target_temp    = 1.0,
)
sim.equilibrate(steps=300)
history = sim.run(steps=500, log_every=50)

# Access results
for state in history:
    print(state.step, state.ke, state.temp)
```

### Sample output

```
Simulation ready: 256 particles | box=25.0 | dt=0.01 | potential=soft | CPU

Equilibrating 300 steps (thermostat ON) …
  Done.  T = 1.0000  KE = 255.0000

  Step      Time          KE          PE      Temp
--------------------------------------------------
   301     3.010    252.5852   1351.7369    0.9905
   360     3.600    255.9235   1348.3744    1.0036
   420     4.200    250.2586   1354.0553    0.9814
   ...
   600     6.000    272.0598   1332.2603    1.0669

Done.  300 steps in 8.7 s (34.6 steps/s)
```

KE fluctuates naturally around `N × T = 256` — the system is in thermal equilibrium.

---

## 🔑 GPU Acceleration Explained

### The kernel (one thread per particle)

```python
@cuda.jit
def _soft_forces_kernel(pos, box, r0sq, A, forces_out):
    i = cuda.grid(1)        # global thread index → particle i
    n = pos.shape[0]

    if i >= n:              # guard: extra threads do nothing
        return

    fx = numba.float32(0.0)
    fy = numba.float32(0.0)

    for j in range(n):      # thread i loops over all j
        if i == j:
            continue
        # ... compute force from j on i ...
        fx += ...
        fy += ...

    forces_out[i, 0] = fx   # one write to global memory per thread
    forces_out[i, 1] = fy
```

| CUDA concept | What it means here |
|---|---|
| `@cuda.jit` | Compile Python → PTX GPU machine code |
| `cuda.grid(1)` | Global thread index across all blocks → particle index |
| `threads_per_block = 128` | Block size (warp-aligned; tune per GPU) |
| `blocks_per_grid = ⌈N/128⌉` | Enough blocks to cover all N particles |
| `cuda.synchronize()` | Block host until all GPU threads finish (required for timing) |

### Why is the GPU faster?

The CPU executes each particle's force computation **sequentially**.  
The GPU executes **all N simultaneously** across thousands of CUDA cores.

For N = 1024 particles:
- CPU: ~1M operations, done one at a time
- GPU: ~1M operations, done in parallel across 1024 threads

---

## 📊 Performance Notes

### Expected speedup (A100 GPU, soft potential)

| N | CPU (ms) | GPU (ms) | Speedup |
|---|---|---|---|
| 64 | ~5 | ~1.5 | ~3× |
| 128 | ~20 | ~1.7 | ~12× |
| 256 | ~80 | ~2.2 | ~36× |
| 512 | ~320 | ~3.5 | ~91× |
| 1024 | ~1280 | ~8.0 | ~160× |

*GPU timings include host↔device memory transfer.*

### When does GPU win?

| N | Regime |
|---|---|
| < 128 | GPU overhead (memory transfer, kernel launch) may dominate |
| 256–512 | GPU starts to pull ahead |
| ≥ 1024 | GPU wins clearly (10–200× depending on hardware) |

The CPU kernel is pure Python — intentionally readable, not optimised.
A NumPy-vectorised CPU would be much faster but less instructive.

---

## 🔭 Future Improvements

| Feature | Difficulty | Impact |
|---|---|---|
| NumPy-vectorised CPU kernel | Easy | 10–100× faster CPU |
| Cell / neighbour lists | Medium | O(N²) → O(N) force calculation |
| CUDA shared memory tiling | Medium | Reduces global memory reads |
| 3-D simulation | Easy | Add z-component throughout |
| Velocity-rescaling thermostat (Berendsen) | Easy | Smoother temperature control |
| Matplotlib / live animation | Easy | Visual output |
| Multiple particle species | Medium | Mixtures, alloys |
| NPT ensemble (pressure control) | Medium | Variable box size |

---

## 📖 References

- Allen & Tildesley, *Computer Simulation of Liquids* (2017) — the MD bible
- Frenkel & Smit, *Understanding Molecular Simulation* (2002)
- [Numba CUDA documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

## 📄 License

MIT — free to use, study, and modify.
