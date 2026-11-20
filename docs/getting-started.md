# Getting Started

## Prerequisites
- Python 3.8 or newer
- **For GPU support:** NVIDIA CUDA Toolkit and a CUDA-capable GPU

## Installation

```bash
pip install numpy numba
```

Or install the package itself (adds the `md_sim` module to your environment):

```bash
pip install -e .
pip install -e ".[test]"   # also installs pytest
```

Numba and the CUDA toolkit are only needed for the GPU path. Without them the
simulator falls back to the CPU implementation automatically.

## Running

```bash
python run_example.py                # 256 particles, soft potential, CPU
python run_example.py --gpu          # CUDA kernel (soft potential only)
python run_example.py --lj           # Lennard-Jones, NVT, CPU only
python run_example.py --n 512        # custom particle count
python run_example.py --steps 1000   # custom run length
python run_example.py --benchmark    # CPU vs GPU timing table
```

## Using the library

```python
from md_sim.simulation import Simulation

sim = Simulation(n_particles=256, box_size=25.0, dt=0.01, use_gpu=True)
sim.equilibrate(steps=300)     # thermostatted
history = sim.run(steps=500)   # NVE production run
```

`potential_type` accepts `'soft'` (default) or `'lj'`; anything else raises a
`ValueError`. The CUDA kernel implements the soft potential only, so `'lj'`
always runs on the CPU.

## Tests

```bash
pytest tests/ -v
```

GPU tests skip automatically when CUDA is unavailable. To exercise the CUDA
kernel without a GPU, run them under Numba's CUDA simulator:

```bash
NUMBA_ENABLE_CUDASIM=1 pytest tests/ -v
```

See the [README](../README.md) for the physics model and benchmark results.
