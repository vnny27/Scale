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
 src/v3_matmul_mma_m32n16k8_swizzle.cu -o matmul_tensor \
 -Xcompiler -fopenmp -lcublas
```

## Run

Run all benchmark versions:

```bash
./scripts/run.sh
```

This builds every source under `src/`, runs each binary with cuBLAS comparison, and writes one combined log plus a compact summary TSV under `results/`.

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
| ref | WMMA global-load baseline | 2.059 | 66755.891 | baseline | 79.7 | Direct `wmma::load_matrix_sync` from global memory |
| v0 | Inline PTX `mma.m16n8k8` baseline | 6.691 | 20542.160 | baseline | 24.5 | One warp computes a `16x8` tile |
| v1 | Logical `16x16x8` MMA tile | 5.087 | 27016.995 | 1.32x | 32.2 | Two `m16n8k8` instructions cover left/right N halves |
| v2.0 | Shared staging without swizzle | 6.030 | 22793.291 | 0.84x | 27.2 | Control version for measuring swizzle impact |
| v2 | Shared staging + XOR swizzle | 3.682 | 37326.250 | 1.64x | 44.5 | Cleaner block/warp order and shared staging |
| v3 | `32x16` warp tile + TF32 preconvert | 2.820 | 48744.406 | 1.31x | 58.1 | Reuse B across two M tiles |
| v4 | `cp.async` double buffering | 2.690 | 51095.000 | 1.05x | 61.3 | 16B async copy over two shared stages |
| v5 | `32x32` warp tile | 2.057 | 66803.034 | 1.31x | 80.0 | Reuse A across four N halves with 2 A fragments and 4 B fragments |

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
- **Result:** `2.059 ms`, `66755.891 GFLOP/s`, `79.7%` of cuBLAS.
- **Takeaway:** WMMA already reaches a strong baseline because it hides most fragment layout and load details behind `load_matrix_sync`. The inline MMA versions start from lower performance, but they expose the exact fragment mapping, shared-memory layout, and pipeline choices that need to be controlled manually.

## Kernel 0 / Inline PTX `m16n8k8` Baseline

- **Goal:** Build a correctness-first inline PTX MMA baseline and understand the lane-level fragment contract.
- **Implementation:** Each warp computes one `16x8` output tile using 4 TF32 A registers, 2 TF32 B registers, and 4 FP32 accumulator registers per lane.
- **Expected effect:** Expose the real TF32 MMA shape and make later shared-memory swizzling easier to reason about.
- **Result:** `6.691 ms`, `20542.160 GFLOP/s`, `24.5%` of cuBLAS.
- **Takeaway:** The first bottleneck is not tensor-core throughput itself, but the small amount of work assigned to each warp. A single native `m16n8k8` tile gives direct control over lane fragments, but it leaves too little operand reuse, so the next step is to compose a larger logical output tile.

## Kernel 1 / Logical `16x16x8` MMA Tile

- **Goal:** Match the familiar `16x16` output tile shape while staying on native `m16n8k8` instructions.
- **Implementation:** Reuse the same A fragment and issue two `m16n8k8` instructions for the left and right `N=8` halves of a logical `16x16` output tile.
- **Expected effect:** Reuse A across two B fragments and reduce per-output control overhead compared with the `16x8` naive tile.
- **Result:** `5.087 ms`, `27016.995 GFLOP/s`, `32.2%` of cuBLAS.
- **Takeaway:** Reusing the same A fragment for two N halves improves arithmetic per fragment load, but the kernel is still mostly limited by repeated global operand fetches and fragment setup. This makes shared-memory staging the natural next step.

## Kernel 2.0 / Shared Staging without Swizzle

- **Goal:** Isolate the effect of shared-memory staging before adding XOR swizzling.
- **Implementation:** Stage A as `shared_a[k][m]` and B as `shared_b[n][k]`, but load/store shared memory with direct indexing.
- **Expected effect:** Provide a control point for measuring how much of Kernel 2's improvement comes from staging alone versus swizzle-based bank-conflict reduction.
- **Result:** `6.030 ms`, `22793.291 GFLOP/s`, `27.2%` of cuBLAS.
- **Takeaway:** Moving operands into shared memory is not enough by itself. The unswizzled layout still maps the MMA fragment load pattern poorly onto shared-memory banks, so the staged operands do not translate into useful throughput. This isolates bank layout as the next bottleneck.

## Kernel 2 / Shared Staging and XOR Swizzle

- **Goal:** Make the shared-memory staging easier to reason about while keeping bank conflicts under control.
- **Implementation:** Stage A as `shared_a[k][m]` and B as `shared_b[n][k]`, use M-fast block/warp tile ordering, and swizzle the fast shared dimension with `A: m ^ ((k & 3) << 3)` and `B: k ^ ((n & 7) << 2)`.
- **Expected effect:** Keep indexing consistent across global, block, warp, and shared-memory levels while still spreading the native `m16n8k8.row.col` operand load pattern across shared-memory banks.
- **Result:** `3.682 ms`, `37326.250 GFLOP/s`, `44.5%` of cuBLAS.
- **Takeaway:** XOR swizzling makes the staged A/B tiles match the lane-level MMA load pattern better, reducing the cost that made v2.0 regress. Once shared-memory staging becomes usable, the remaining opportunity is to make each loaded fragment feed more MMA work.

## Kernel 3 / `32x16` Warp Tile and Shared TF32 Preconvert

- **Goal:** Increase useful fragment reuse and reduce repeated TF32 conversion work in the MMA hot loop.
- **Implementation:** Let one warp compute a logical `32x16` output tile by using two A fragments for the upper/lower M halves and reusing the same B left/right fragments. Shared memory now stages A as `shared_a[k][m]` and B as `shared_b[n][k]`, stores TF32 bit patterns as `unsigned int`, and uses the same M-fast block/warp tile ordering as Kernel 2.
- **Expected effect:** Reuse B across two M tiles, remove repeated conversion instructions from the inner loop, and keep the indexing consistent with Kernel 2.
- **Result:** `2.820 ms`, `48744.406 GFLOP/s`, `58.1%` of cuBLAS.
- **Takeaway:** The `32x16` warp tile improves reuse by letting one B fragment feed two M halves, similar to how CUDA-core micro-tiling increases work per loaded value. Preconverting TF32 values also removes repeated conversion from the hot loop, so the next bottleneck moves toward the global-to-shared copy path.

## Kernel 4 / `cp.async` Double Buffering

- **Goal:** Hide part of the global-to-shared copy latency by preloading the next `K` tile while computing the current tile.
- **Implementation:** Use two shared-memory stages for A/B and issue 16B `cp.async` copies for the next `BLOCK_K` tile before the current tile's MMA loop runs.
- **Expected effect:** Overlap global memory traffic with tensor-core compute and reduce `cp.async` instruction overhead compared with scalar 4B copies.
- **Result:** `2.690 ms`, `51095.000 GFLOP/s`, `61.3%` of cuBLAS.
- **Takeaway:** Double buffering with `cp.async` improves the copy path by overlapping the next K-tile load with current-tile MMA work. The gain is modest because the extra shared-memory stage and pipeline bookkeeping are large relative to the work in a `32x16` warp tile, which motivates increasing the warp tile size.

## Kernel 5 / `32x32` Warp Tile

- **Goal:** Amortize the double-buffering and fragment-load overhead by increasing the work owned by each warp.
- **Implementation:** Expand the warp tile from `32x16` to `32x32`. Each warp now uses two A fragments and four B fragments, issuing eight native `m16n8k8` MMA instructions per `K=8` slice.
- **Expected effect:** Reuse the same A fragments across four N halves and reduce per-output scheduling overhead, while accepting higher accumulator register pressure.
- **Result:** `2.057 ms`, `66803.034 GFLOP/s`, `80.0%` of cuBLAS.
- **Takeaway:** The `32x32` warp tile mainly improves register reuse of A fragments: each loaded A fragment is reused across four `N=8` MMA slices instead of two. Since each block also covers twice as many output columns, the same A shared tile and async pipeline steps feed more MMA work. This recovers WMMA-level performance.

## Future Work
