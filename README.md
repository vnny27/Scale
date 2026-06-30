# CUDA Matmul Experiments

This repository collects CUDA matrix multiplication experiments, from CUDA-core SGEMM kernels to WMMA / TF32 tensor-core kernels.

The main goal is to keep each optimization step measurable: every kernel variant should have a clear hypothesis, benchmark result, and profiling note when useful.

## Directory Structure

| Path | Description |
| --- | --- |
| `matmul/` | CUDA-core SGEMM optimization experiments. Includes naive, tiled, register-tiled, vectorized, double-buffered, and `cp.async` versions. |
| `matmul/README.md` | Detailed CUDA-core benchmark notes, optimization log, summary table, and shape experiments. |
| `matmul_tensor/` | Tensor-core matmul experiment using WMMA with TF32 tensor core compute. |
| `matmul_tensor/README.md` | Tensor-core benchmark notes and current profiling summary. |

## Projects

### CUDA-Core SGEMM

See `matmul/`.

This project tracks a step-by-step CUDA-core SGEMM optimization path:

- global-memory baseline
- shared-memory tiling
- per-thread micro tiling
- shared-memory layout experiments
- column-major layout
- warp tiling and lane mapping
- vectorized memory access
- double buffering
- `cp.async` prefetching
- tall/wide shape experiments

### Tensor-Core SGEMM

See `matmul_tensor/`.

This project currently has WMMA kernel:

- `float` input matrices
- TF32 tensor core computation
- FP32 accumulation
- column-major matrix layout
- `128x128x32` CTA tile
- `16x16x8` WMMA fragments

## Typical Workflow

For each experiment:

1. Implement or adjust a kernel variant.
2. Run correctness checks.
3. Compare against cuBLAS when relevant.
4. Profile with Nsight Compute or Nsight Systems.
5. Record the result and takeaway in the project README.