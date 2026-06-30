# CUDA Tensor Core Matmul Benchmark Notes

This directory tracks experiments for optimizing `matmul_tensor.cu` with WMMA / TF32 tensor cores and comparing it with cuBLAS SGEMM.

## Environment

Fill this in for each benchmark session.

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| Driver version | TBD |
| CUDA version | TBD |
| Matrix size | `M=N=K=4096` |
| Data type | `float` input, TF32 tensor core compute |
| Warmup iterations | `2` |
| Benchmark iterations | `10` |

## Build

Basic benchmark with cuBLAS comparison:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -DUSE_CUBLAS matmul_tensor.cu -o matmul_tensor -Xcompiler -fopenmp -lcublas
```

With NVTX ranges for Nsight Systems / Nsight Compute:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 \
 -DUSE_CUBLAS -DUSE_NVTX -I \
 /usr/local/cuda-13.1/targets/x86_64-linux/include/nvtx3 \
 matmul_tensor.cu -o matmul_tensor -Xcompiler -fopenmp -lcublas
```

## Run

Correctness check:

```bash
./matmul_tensor --verify
```

Compare custom kernel with cuBLAS:

```bash
./matmul_tensor --cublas
```

Compare and verify both outputs:

```bash
./matmul_tensor --verify --cublas
```

## Profiling

Nsight Systems:

```bash
nsys profile --trace=cuda,nvtx,cublas --stats=true -o tensor_matmul_nsys ./matmul_tensor --cublas
nsys stats --force-export=true --report \
 nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum tensor_matmul_nsys.nsys-rep
```

Nsight Compute, custom kernel:

```bash
/usr/local/cuda-13.1/bin/ncu --set full --target-processes all --nvtx --nvtx-include \
 "custom_matmul/" -o tensor_full ./matmul_tensor --cublas
```

Nsight Compute, cuBLAS range:

```bash
/usr/local/cuda-13.1/bin/ncu --set full --target-processes all --nvtx --nvtx-include \
 "cublas_sgemm/" -o tensor_cublas ./matmul_tensor --cublas
```

## Summary Results

The first row uses the current WMMA kernel as the baseline. The time below is from `tensor_full_32k.ncu-rep` and should be treated as a profiled kernel duration, not a normal benchmark-loop timing.

| Version | Main change | Custom time (ms) | Custom GFLOP/s | Speedup vs previous | cuBLAS (%) | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| v0 | WMMA TF32 tensor core kernel | 3.203 | 42902.909 | baseline | TBD | `128x128x32` CTA tile, `4x4` warps, `16x16x8` WMMA tiles |

## Optimization Log

Use the same structure for each kernel note:

- **Goal:** What bottleneck or hypothesis this version targets.
- **Implementation:** The concrete code-level change.
- **Expected effect:** Why this should improve throughput, memory behavior, occupancy, or instruction scheduling.
- **Result:** Benchmark or profiling numbers from the summary table.
- **Takeaway:** What this version teaches and what it motivates next.

## Kernel 0 / Initial WMMA Tensor Core Kernel

- **Goal:** Establish a tensor-core baseline using WMMA with TF32 inputs and FP32 accumulation.
- **Implementation:** Use a `128x128` output tile per CTA, `BLOCK_K=32`, `4x4` warps per block, and `16x16x8` WMMA fragments. `A`, `B`, and `C` are stored in column-major order. Each K tile is staged through shared memory before `wmma::load_matrix_sync`, `wmma::mma_sync`, and `wmma::store_matrix_sync`.
- **Expected effect:** Move the main compute work from CUDA cores to tensor cores, greatly increasing peak FMA throughput compared with scalar FP32 FMA kernels.
- **Result:** `3.203 ms`, `42902.909 GFLOP/s`
- **Takeaway:** The kernel successfully uses tensor cores, but the profile still shows relatively low achieved occupancy and issue utilization. The current structure is a useful correctness and profiling baseline before adding pipeline improvements such as double buffering, `cp.async`, or different warp/block shapes.

## Future Work
