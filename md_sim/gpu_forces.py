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

import numpy as np

# ── detect CUDA availability ─────────────────────────────────────────────────
try:
    from numba import cuda
    import numba
    _CUDA_AVAILABLE = cuda.is_available()
except ImportError:
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
    def _soft_forces_kernel(pos, box, r0sq, A, forces_out):
        """
        GPU kernel: one thread → one particle i.

        Parameters (all device arrays / scalars)
        ----------
        pos        : float32[N, 2]  particle positions
        box        : float32        box side length
        r0sq       : float32        interaction range squared (r₀²)
        A          : float32        repulsion strength
        forces_out : float32[N, 2]  output forces (written once per thread)
        """
        i = cuda.grid(1)     # global thread index == particle i
        n = pos.shape[0]

        if i >= n:           # guard: extra threads do nothing
            return

        fx = numba.float32(0.0)
        fy = numba.float32(0.0)

        xi = pos[i, 0]
        yi = pos[i, 1]

        for j in range(n):
            if i == j:
                continue

            # ── minimum-image displacement ───────────────────────────────────
            dx = pos[j, 0] - xi
            dy = pos[j, 1] - yi

            half_box = box * numba.float32(0.5)
            if dx >  half_box: dx -= box
            if dx < -half_box: dx += box
            if dy >  half_box: dy -= box
            if dy < -half_box: dy += box

            r2 = dx * dx + dy * dy

            # ── soft repulsion ───────────────────────────────────────────────
            if r2 < r0sq and r2 > numba.float32(1e-14):
                r       = r2 ** numba.float32(0.5)
                r0      = r0sq ** numba.float32(0.5)
                overlap = numba.float32(1.0) - r / r0

                # |F| / r = 2A * overlap / (r0 * r²)  →  F_vec = |F|/r * dr
                f_over_r = numba.float32(2.0) * A * overlap / (r0 * r2)

                fx -= f_over_r * dx   # repel i away from j
                fy -= f_over_r * dy

        # write result to global memory (one write per thread)
        forces_out[i, 0] = fx
        forces_out[i, 1] = fy

    return _soft_forces_kernel


if _CUDA_AVAILABLE:
    _soft_forces_kernel = _define_cuda_kernel()
else:
    _soft_forces_kernel = None


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
    """
    if not _CUDA_AVAILABLE:
        raise RuntimeError(
            "Numba CUDA is not available. Use use_gpu=False for the CPU path."
        )

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
        np.float32(r0 ** 2),
        np.float32(A),
        d_forces,
    )
    cuda.synchronize()   # wait for all GPU threads before returning

    # copy forces back to host
    forces = d_forces.copy_to_host()
    return forces, None
