"""
gpu_forces.py — GPU force kernel (Numba CUDA)
==============================================
Parallel soft-repulsion force calculation where *each thread handles
one particle i* and loops over all j.

The potential used here matches cpu_forces.py's 'soft' mode:

    U(r) = A·(1 − r/r₀)²   for  r < r₀,  else  0

    Force on i from j:
    F_ij = −2A/r₀ · (1 − r/r₀) · r̂_ij   (outward = repulsive)

This "one thread per atom" pattern is the simplest possible CUDA
decomposition — ideal for learning GPU programming.

Thread indexing
---------------
    grid  →  blocks  →  threads
    cuda.grid(1) returns the global thread index across all blocks.
    We map: global_thread_index  →  particle_index_i
"""

from __future__ import annotations   # PEP 585 annotations below; see cpu_forces

import numpy as np

# ── detect CUDA availability ─────────────────────────────────────────────────
# Any failure here (numba missing, broken CUDA driver, unsupported numpy ABI)
# must degrade to the CPU path rather than break `import md_sim`.
try:
    from numba import cuda
    import numba
    _CUDA_AVAILABLE = bool(cuda.is_available())
except Exception:
    _CUDA_AVAILABLE = False
    cuda   = None
    numba  = None


# ── define kernel (only when CUDA is present) ────────────────────────────────

def _define_cuda_kernel():
    """
    Compile and return the CUDA kernel.
    Wrapped in a function so @cuda.jit only runs on GPU machines.
    """

    @cuda.jit
    def _soft_forces_kernel(pos, box, r0, A, forces_out):
        """
        GPU kernel: one thread → one particle i.

        Parameters (all device arrays / scalars)
        ----------
        pos        : float32[N, 2]  particle positions
        box        : float32        box side length
        r0          : float32       interaction diameter (r₀)
        A          : float32        repulsion strength
        forces_out : float32[N, 2]  output forces (written once per thread)

        r₀ is passed rather than r₀² so the device never computes
        sqrt(float32(r0**2)), which can differ from the host's r₀ by an ulp
        and put a systematic offset into every GPU/CPU comparison.
        """
        i = cuda.grid(1)     # global thread index == particle i
        n = pos.shape[0]

        if i >= n:           # guard: extra threads do nothing
            return

        fx = numba.float32(0.0)
        fy = numba.float32(0.0)

        xi = pos[i, 0]
        yi = pos[i, 1]

        # Loop-invariant quantities: compute once per thread, not per pair.
        r0sq     = r0 * r0
        half_box = box * numba.float32(0.5)

        for j in range(n):
            if i == j:
                continue

            # ── minimum-image displacement ───────────────────────────────────
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi

            if dx >  half_box: dx -= box
            if dx < -half_box: dx += box
            if dy >  half_box: dy -= box
            if dy < -half_box: dy += box

            r2 = dx * dx + dy * dy

            # ── soft repulsion ───────────────────────────────────────────────
            if r2 < r0sq and r2 > numba.float32(1e-14):
                r       = r2 ** numba.float32(0.5)
                overlap = numba.float32(1.0) - r / r0

                # |F| = 2A * overlap / r0, so  |F| / r = 2A * overlap / (r0 * r)
                # and  F_vec = (|F| / r) * dr.  Dividing by r2 here instead of r
                # would scale every pair force by a spurious 1 / r.
                f_over_r = numba.float32(2.0) * A * overlap / (r0 * r)

                fx -= f_over_r * dx   # repel i away from j
                fy -= f_over_r * dy

        # write result to global memory (one write per thread)
        forces_out[i, 0] = fx
        forces_out[i, 1] = fy

    return _soft_forces_kernel


_soft_forces_kernel = None
if _CUDA_AVAILABLE:
    # @cuda.jit decoration can still fail even when cuda.is_available() is
    # True (numba/numpy ABI mismatch, target init errors).  Degrade to the CPU
    # path instead of letting `import md_sim` die.
    try:
        _soft_forces_kernel = _define_cuda_kernel()
    except Exception as exc:        # pragma: no cover - needs a broken CUDA stack
        print(f"[Warning] CUDA kernel compilation failed ({exc}); using CPU.")
        _CUDA_AVAILABLE = False


# ── host-side wrapper ─────────────────────────────────────────────────────────

def compute_forces_gpu(
    positions: np.ndarray,
    box_size:  float,
    r0:        float = 2.0,
    A:         float = 50.0,
) -> tuple[np.ndarray, None]:
    """
    Host wrapper: copies data to GPU, launches the kernel, copies back.

    Parameters
    ----------
    positions : ndarray (N, 2), float32
    box_size  : float
    r0        : float  — soft potential interaction diameter
    A         : float  — soft potential repulsion strength

    Returns
    -------
    forces : ndarray (N, 2), float32
    None   — potential energy not computed on GPU (would need atomic adds)

    Preconditions
    -------------
    `positions` must already lie inside [0, box_size).  The kernel applies a
    single minimum-image shift per axis (one ±box correction), which is exact
    for in-box coordinates but — unlike the CPU path's round()-based fold —
    does not wrap offsets beyond 1.5·box.  Feeding it unwrapped coordinates
    silently drops interactions rather than raising.  `Simulation` wraps every
    step (integrator.velocity_verlet_step1), so this always holds there.
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError(
            "Numba CUDA is not available. Use use_gpu=False for the CPU path."
        )

    # The kernel's constants and the output array are float32; make sure the
    # input matches so the kernel is not recompiled for another dtype.
    positions = np.ascontiguousarray(positions, dtype=np.float32)
    n = positions.shape[0]

    # copy positions to GPU
    d_pos    = cuda.to_device(positions)
    d_forces = cuda.device_array((n, 2), dtype=np.float32)

    # launch config: 128 threads/block is a common starting point
    threads_per_block = 128
    blocks_per_grid   = (n + threads_per_block - 1) // threads_per_block

    # launch kernel
    _soft_forces_kernel[blocks_per_grid, threads_per_block](
        d_pos,
        np.float32(box_size),
        np.float32(r0),
        np.float32(A),
        d_forces,
    )
    cuda.synchronize()   # wait for all GPU threads before returning

    # copy forces back to host
    forces = d_forces.copy_to_host()
    return forces, None
