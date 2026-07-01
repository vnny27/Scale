# CUDA Tensor Core Matmul Benchmark Notes

This directory tracks experiments for optimizing TF32 tensor-core matrix multiplication with both WMMA and inline PTX MMA, and compares the custom kernels with cuBLAS SGEMM.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| Driver version | 590.48.01 |
| CUDA version | 13.1 |
| Matrix size | `M=N=K=4096` |
| Data type | `float` input, TF32 tensor core compute, FP32 accumulation |
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

To test another experiment, replace `matmul_tensor.cu` with the target source file, for example:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 -DUSE_CUBLAS \
 matmul_tensor_mma_m32n16k16_swizzle.cu -o matmul_tensor \
 -Xcompiler -fopenmp -lcublas
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

| Version | Main change | Custom time (ms) | Custom GFLOP/s | Speedup vs previous | cuBLAS (%) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v0 | WMMA global-load baseline | 2.059 | 66755.891 | baseline | 79.7 | Direct `wmma::load_matrix_sync` from global memory |
| v0 | Inline PTX `mma.m16n8k8` baseline | 6.691 | 20542.160 | baseline | 24.5 | One warp computes a `16x8` tile |
| v1 | Logical `16x16x8` MMA tile | 5.087 | 27016.995 | 1.32x | 32.2 | Two `m16n8k8` instructions cover left/right N halves |
| v2 | Shared staging + XOR swizzle | 3.682 | 37326.250 | 1.38x | 44.5 | Cleaner block/warp order and shared staging |
| v3 | `32x16` warp tile + TF32 preconvert | 2.820 | 48744.406 | TBD | 58.1 | TBD |

## Optimization Log

Use the same structure for each kernel note:

- **Goal:** What bottleneck or hypothesis this version targets.
- **Implementation:** The concrete code-level change.
- **Expected effect:** Why this should improve throughput, memory behavior, occupancy, or instruction scheduling.
- **Result:** Benchmark or profiling numbers from the summary table.
- **Takeaway:** What this version teaches and what it motivates next.

## Reference / WMMA Global-Load Baseline

- **Goal:** Establish a high-level tensor-core reference before moving to inline PTX MMA.
- **Implementation:** Each warp owns a sub-tile of a `128x128` block tile and directly calls `wmma::load_matrix_sync` on global-memory `A` and `B`.
- **Expected effect:** Use tensor cores with minimal manual lane mapping or shared-memory layout work.
- **Result:** `2.059 ms`, `66744.891 GFLOP/s`, `79.7%` of cuBLAS.
- **Takeaway:** WMMA provides a strong reference point with much less manual code. The MMA experiments below are mainly about understanding and controlling the lower-level fragment layout.

## Kernel 0 / Inline PTX `m16n8k8` Baseline

- **Goal:** Build a correctness-first inline PTX MMA baseline and understand the lane-level fragment contract.
- **Implementation:** Each warp computes one `16x8` output tile using 4 TF32 A registers, 2 TF32 B registers, and 4 FP32 accumulator registers per lane.
- **Expected effect:** Expose the real TF32 MMA shape and make later shared-memory swizzling easier to reason about.
- **Result:** `6.691 ms`, `20542.160 GFLOP/s`, `24.5%` of cuBLAS.
- **Takeaway:** TF32 MMA has native `m16n8k8`, not native `m16n16k8`. This baseline is slower than WMMA but it gives direct control over lane mapping, register fragments, and future shared-memory layout.

## Kernel 1 / Logical `16x16x8` MMA Tile

- **Goal:** Match the familiar `16x16` output tile shape while staying on native `m16n8k8` instructions.
- **Implementation:** Reuse the same A fragment and issue two `m16n8k8` instructions for the left and right `N=8` halves of a logical `16x16` output tile.
- **Expected effect:** Reuse A across two B fragments and reduce per-output control overhead compared with the `16x8` naive tile.
- **Result:** `5.087 ms`, `27016.995 GFLOP/s`, `32.2%` of cuBLAS.
- **Takeaway:** The logical `16x16` construction is correct but simply adding the second N half increased register/instruction pressure without adding shared-memory reuse. This motivated staging A/B tiles in shared memory and focusing on bank layout.

## Kernel 2 / Shared Staging and XOR Swizzle

- **Goal:** Make the shared-memory staging easier to reason about while keeping bank conflicts under control.
- **Implementation:** Stage A as `shared_a[k][m]` and B as `shared_b[n][k]`, use M-fast block/warp tile ordering, and swizzle the fast shared dimension with `A: m ^ ((k & 3) << 3)` and `B: k ^ ((n & 7) << 2)`.
- **Expected effect:** Keep indexing consistent across global, block, warp, and shared-memory levels while still spreading the native `m16n8k8.row.col` operand load pattern across shared-memory banks.
- **Result:** `3.682 ms`, `37326.250 GFLOP/s`, `44.5%` of cuBLAS.
- **Takeaway:** This version makes the layout story cleaner by keeping block/warp traversal and shared-memory staging consistent. It keeps the lightweight swizzle idea while avoiding the earlier `[row][col]` staging ambiguity.

## Kernel 3 / `32x16` Warp Tile and Shared TF32 Preconvert

- **Goal:** Increase useful fragment reuse and reduce repeated TF32 conversion work in the MMA hot loop.
- **Implementation:** Let one warp compute a logical `32x16` output tile by using two A fragments for the upper/lower M halves and reusing the same B left/right fragments. Shared memory now stages A as `shared_a[k][m]` and B as `shared_b[n][k]`, stores TF32 bit patterns as `unsigned int`, and uses the same M-fast block/warp tile ordering as Kernel 2.
- **Expected effect:** Reuse B across two M tiles, remove repeated conversion instructions from the inner loop, and keep the indexing consistent with Kernel 2.
- **Result:** TBD after rebenchmark.
- **Takeaway:** This version preserves the main Kernel 3 idea, but uses the cleaner shared staging from Kernel 2. The next measurement should show whether this keeps the previous reuse benefit without reintroducing shared-memory conflicts.

## Future Work

- Add `cp.async` only after the MMA shared-memory layout has enough per-tile compute reuse to hide copy overhead.
