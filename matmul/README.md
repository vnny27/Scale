# CUDA Matmul Benchmark Notes

This directory tracks experiments for optimizing `matmul.cu` and comparing it with cuBLAS SGEMM.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090  |
| Driver version | 590.48.01 |
| CUDA version | 13.1 |
| Matrix size | `M=N=K=4096` |
| Data type | `float` |
| Warmup iterations | `10` |
| Benchmark iterations | `30` |

## Build

Basic benchmark with cuBLAS comparison:

```bash
/usr/local/cuda/bin/nvcc -O3 -DUSE_CUBLAS \
 matmul.cu -o matmul -Xcompiler -fopenmp -lcublas
```

With NVTX ranges for Nsight Systems / Nsight Compute:

```bash
/usr/local/cuda/bin/nvcc -O3 -arch=sm_89 \
 -DUSE_CUBLAS -DUSE_NVTX -I \
 /usr/local/cuda-13.1/targets/x86_64-linux/include/nvtx3 \
 matmul.cu -o matmul -Xcompiler -fopenmp -lcublas
```

## Run

Run all benchmark versions:

```bash
./scripts/run.sh
```

This builds every source under `src/`, runs each binary with cuBLAS comparison, and writes one combined log plus a compact summary TSV under `results/`.

Correctness check:

```bash
./matmul --verify
```

Compare custom kernel with cuBLAS:

```bash
./matmul --cublas
```

Compare and verify both outputs:

```bash
./matmul --verify --cublas
```

## Profiling

Nsight Systems:

```bash
nsys profile --trace=cuda,nvtx,cublas --stats=true -o matmul_nsys ./matmul --cublas
nsys stats --force-export=true --report \ 
 nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum matmul_nsys.nsys-rep
```

Nsight Compute, custom kernel:

```bash
/usr/local/cuda-13.1/bin/ncu --set full --target-processes all --nvtx --nvtx-include \
 "custom_matmul/" -o custom_matmul_ncu ./matmul --cublas
```

Nsight Compute, cuBLAS range:

```bash
/usr/local/cuda-13.1/bin/ncu --set full --target-processes all --nvtx --nvtx-include \
 "cublas_sgemm/" -o cublas_sgemm_ncu ./matmul --cublas
```

## Summary Results

| Version | Main change | Custom time (ms) | Custom GFLOP/s | Speedup vs previous | cuBLAS (%) | Notes |
| --- | --- | ---: | ---: | ---: | --- | --- |
| v0 | naive version | 135.800 | 1012.070 | baseline | 2.2 |  |
| v1 | tiled kernel | 34.850 | 3943.765 | 3.90x | 6.6 |  |
| v2 | 256 threads/block, 4x4 thread tile | 4.132 | 33259 | 8.43x | 55.6 |  |
| v3 | interleaved B | 4.382 | 31362.947 | 0.94x | 52.4 | Regressed; <br> motivated transpose B |
| v4 | transpose B shared layout | 3.773 | 36424.729 | 1.16x | 60.3 |  |
| v5 | warp tiling | 3.728 | 36867.842 | 1.01x | 60.8 |  |
| v6 | vectorized load/store + larger tile | 3.242 | 42393.469 | 1.15x | 70.9 |  |
| v7 | factorize compute | 3.097 | 44372.431 | 1.05x | 74.2 |  |
| v8 | lane mapping | 2.964 | 46376.830 | 1.04x | 77.5 |  |
| v9 | double buffering | 2.551 | 53877.143 | 1.16x | 88.7 |  |
| v10 | add cp.async | 2.456 | 55952.032 | 1.04x | 93.8 |  |

## Optimization Log

Use the same structure for each kernel note:

- **Goal:** What bottleneck or hypothesis this version targets.
- **Implementation:** The concrete code-level change.
- **Expected effect:** Why this should improve throughput, memory behavior, occupancy, or instruction scheduling.
- **Result:** Benchmark numbers from the summary table.
- **Takeaway:** What this version teaches and what it motivates next.

## Kernel 0 / Naive Global-Memory Kernel

- **Goal:** Establish a correctness-first baseline.
- **Implementation:** One thread computes one `C(row, col)` element using a full `K` loop over global memory.
- **Expected effect:** Simple reference point; expected to be memory-inefficient because each thread repeatedly loads from global memory with little reuse.
- **Result:** `135.800 ms`, `1012.070 GFLOP/s`, `2.2%` of cuBLAS.
- **Takeaway:** The main bottleneck is poor memory reuse. Neighboring threads repeatedly load the same `A` and `B` values from global memory, so the first useful optimization is to stage reusable tiles closer to the SM.

## Kernel 1 / Shared-Memory Tiling

- **Goal:** Reduce redundant global memory traffic by reusing `A` and `B` tiles inside a thread block.
- **Implementation:** Load `32x32` tiles into shared memory and accumulate one output element per thread across `BLOCK_K`.
- **Expected effect:** Higher arithmetic intensity through tile reuse, with synchronization between K-tiles.
- **Result:** `34.850 ms`, `3943.765 GFLOP/s`, `6.6%` of cuBLAS.
- **Takeaway:** Shared memory greatly improves reuse of `A` and `B` across threads, which explains the large speedup over v0. However, each thread still computes only one `C` element, so each shared-memory load feeds too little arithmetic work; register tiling is the next step.

## Kernel 2 / 4x4 Thread Tile

- **Goal:** Increase per-thread work and register reuse.
- **Implementation:** Use a `64x64` output tile with `256` threads per block; each thread computes a `4x4` micro-tile.
- **Expected effect:** More FMAs per global/shared-memory load and better amortization of address calculation and synchronization.
- **Result:** `4.132 ms`, `33259 GFLOP/s`, `55.6%` of cuBLAS.
- **Takeaway:** The `4x4` micro-tile lets each thread reuse loaded `A` values across several columns and loaded `B` values across several rows. This turns the inner loop into a small register-level outer product and is the first major jump in arithmetic intensity.

## Kernel 3 / Interleaved B Layout

- **Goal:** Make the row-major `B`/`C` column access pattern more regular across the warp.
- **Implementation:** Change the per-thread output columns from contiguous columns `tx * THREAD_TILE + j` to interleaved columns `tx + j * BLOCK_THREADS`, and store `B` in shared memory as `subtileB[k][tx][j]` with padding.
- **Expected effect:** Let threads with the same `j` access neighboring logical columns, improving warp-level regularity for `B` reads and `C` writes while reducing shared-memory bank conflicts.
- **Result:** `4.382 ms`, `31362.947 GFLOP/s`, `52.4%` of cuBLAS.
- **Takeaway:** The idea was reasonable for the row-major kernel: instead of making each thread own four contiguous columns, distribute those columns across the warp so a fixed `j` step maps to neighboring output columns. However, this was only a partial reordering of the `B` tile: it made the warp-level column pattern more regular, but the compute loop still did not get a simple `k`-fixed, column-contiguous view of `B`. This regression made the next step clearer: rather than layering another fix onto the row-major/interleaved layout, reorganize `B` in shared memory around the way the inner outer-product loop actually consumes it.

## Kernel 4 / Transposed Shared B

- **Goal:** Align the shared `B` layout with the way the inner outer-product loop consumes `B`.
- **Implementation:** Repack `B` in shared memory with a transposed layout so each fixed-`k` compute step can read the needed columns in a regular layout.
- **Expected effect:** Better shared-memory locality and fewer bank conflicts than the previous `B` layout.
- **Result:** `3.773 ms`, `36424.729 GFLOP/s`, `60.3%` of cuBLAS.
- **Takeaway:** The benefit comes from matching shared `B` layout to the way the compute loop reads `B`. Transposing/reordering `B` in shared memory makes the inner-loop reads more regular and recovers the v3 regression.

## Kernel 5 / Warp Tiling

- **Goal:** Give each warp a clear sub-tile of the thread block output tile.
- **Implementation:** Partition the block tile into warp tiles using `WARPS_M`, `WARPS_N`, lane-to-row, and lane-to-column mapping.
- **Expected effect:** More predictable per-warp memory access and better control over register-tile placement.
- **Result:** `3.728 ms`, `36867.842 GFLOP/s`, `60.8%` of cuBLAS.
- **Takeaway:** Warp-level tiling barely changes the raw compute pattern, so the immediate speedup is small. Its value is structural: it maps the output tile onto hardware execution units more explicitly, which makes later lane mapping and vectorization easier to reason about.

## Kernel 6 / Vectorized Load/Store and Larger Tile

- **Goal:** Increase memory transaction efficiency and scale the block tile.
- **Implementation:** Move to a `128x128` output tile with `8x8` thread tiles, `float4` vectorized loads/stores, and padded shared memory.
- **Expected effect:** Better global-memory coalescing, fewer instructions per loaded element, and more data reuse per block.
- **Result:** `3.242 ms`, `42393.469 GFLOP/s`, `70.9%` of cuBLAS.
- **Takeaway:** The larger tile increases reuse per thread block, while `float4` loads/stores reduce scalar memory instructions and align the data movement with wider memory transactions. The tradeoff is higher register and shared-memory pressure, so later gains require better scheduling and pipelining.

## Kernel 7 / Factorized Compute

- **Goal:** Reduce inner-loop instruction overhead and expose more independent FMAs.
- **Implementation:** Factor repeated compute patterns into a tighter accumulation structure.
- **Expected effect:** Improve instruction scheduling and reduce repeated indexing or scalar bookkeeping.
- **Result:** `3.097 ms`, `44372.431 GFLOP/s`, `74.2%` of cuBLAS.
- **Takeaway:** Simplifying the compute body reduces repeated indexing and gives the compiler a tighter FMA pattern to schedule. The gain is modest, but useful once the larger memory-reuse problems have already been addressed.

## Kernel 8 / Lane Mapping

- **Goal:** Tune how lanes map to output rows and columns inside each warp tile.
- **Implementation:** Adjust lane decomposition so each warp's memory and compute pattern better matches the `8x8` thread tile.
- **Expected effect:** Improve coalescing, shared-memory access regularity, and scheduler efficiency.
- **Result:** `2.964 ms`, `46376.830 GFLOP/s`, `77.5%` of cuBLAS.
- **Takeaway:** Lane mapping changed the warp's traversal order from column-first to row-first inside each warp tile. This made neighboring lanes cover different row bands within the same column group, which fits the later shared-memory and vectorized access pattern better.

## Kernel 9 / Double Buffering

- **Goal:** Overlap global-memory loading for the next K-tile with compute on the current K-tile.
- **Implementation:** Use two shared-memory stages and alternate read/write buffers across the K loop.
- **Expected effect:** Hide part of global-memory latency and keep the compute pipeline busier.
- **Result:** `2.551 ms`, `53877.143 GFLOP/s`, `88.7%` of cuBLAS.
- **Takeaway:** Double buffering overlaps loading the next K-tile with computing the current one, reducing exposed memory latency and pushing the kernel much closer to cuBLAS.

## Kernel 10 / `cp.async` Prefetch

- **Goal:** Use asynchronous global-to-shared copies to improve the double-buffered pipeline.
- **Implementation:** Replace regular `A` tile loads with `cp.async` into shared memory, then commit/wait around the staged pipeline; keep `B` prefetching in registers before shared-memory stores.
- **Expected effect:** Reduce blocking load latency and make the copy/compute pipeline more explicit.
- **Result:** `2.456 ms`, `55952.032 GFLOP/s`, `93.8%` of cuBLAS.
- **Takeaway:** `cp.async` makes the staged pipeline more explicit by asynchronously copying the next `A` tile into shared memory while computation continues so that it exploits the utility of double buffering. 

## Future Work

- Add tall/wide shape benchmarks. The current kernel assumes `M` and `N` are multiples of `128`, and `K` is a multiple of `16`.

| Shape class | M | N | K | Custom time (ms) | Custom GFLOP/s | cuBLAS (%) | Purpose |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline square | 4096 | 4096 | 4096 | 2.456 | 55952.032 | 93.8 | Reference point for existing results |
| Tall output, deep K | 16384 | 512 | 4096 | 1.360 | 50528.455 | 89.2 | Test large-row/small-column output against the current tiled kernel |
| Taller output, deep K | 32768 | 512 | 4096 | 2.754 | 49908.289 | 88.6 | Check whether the tall-output trend holds with more row blocks |
| Tall output, wider N | 16384 | 1024 | 4096 | 2.778 | 49474.318 | 87.5 | Separate tall-shape behavior from the extreme `N=512` case |
| Wide output, deep K | 512 | 16384 | 4096 | 1.327 | 51792.321 | 94.0 | Contrast against tall output with the same output element count |
| Wider output, deep K | 512 | 32768 | 4096 | 2.636 | 52144.525 | 94.1 | Check whether increasing `N` further hurts the current tiled kernel |
| Wide output, taller M | 1024 | 16384 | 4096 | 2.594 | 52986.189 | 93.9 | Check whether adding more row blocks improves wide-output utilization |
| Tall output, shallow K | 16384 | 512 | 512 | 0.171 | 50226.112 | 100.6 | Measure the effect of lower arithmetic intensity |
| Wide output, shallow K | 512 | 16384 | 512 | 0.171 | 50252.251 | 100.5 | Compare shallow-K behavior against the tall case |
