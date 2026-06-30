#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <math.h>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <random>
#include <algorithm>
#include <string>
#include <iomanip>

#include </usr/local/cuda/include/cuda.h>
#include </usr/local/cuda/include/cuda_runtime_api.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif
#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

struct VersionConfig {
    long int M;
    long int K;
    long int N;
};

#define SEED 1234
#define WARMUP_ITERATIONS 10
#define BENCHMARK_ITERATIONS 30


#define MATRIX_M 4096
#define MATRIX_N 4096
#define MATRIX_K 4096
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 16
#define THREAD_TILE_M 8
#define THREAD_TILE_N 8
#define WARPS_M 4
#define WARPS_N 2
#define WARP_THREADS 32
#define WARP_TILE_M (BLOCK_M / WARPS_M)
#define WARP_TILE_N (BLOCK_N / WARPS_N)
#define LANES_M (WARP_TILE_M / THREAD_TILE_M)
#define LANES_N (WARP_TILE_N / THREAD_TILE_N)
#define BLOCK_THREADS (WARPS_M * WARPS_N * WARP_THREADS)
#define FLOAT4_WIDTH 4
#define PREFETCH_VECS_PER_THREAD (BLOCK_K / 8)

__device__ __forceinline__ void cp_async_float4(void *dst, const void *src) {
    unsigned int smem_addr =
        static_cast<unsigned int>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                     "r"(smem_addr), "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}

#define FMA_ACCUM_COL(a0, a1, b, c0, c1) \
    do { \
        (c0).x = fmaf((a0).x, (b), (c0).x); \
        (c0).y = fmaf((a0).y, (b), (c0).y); \
        (c0).z = fmaf((a0).z, (b), (c0).z); \
        (c0).w = fmaf((a0).w, (b), (c0).w); \
        (c1).x = fmaf((a1).x, (b), (c1).x); \
        (c1).y = fmaf((a1).y, (b), (c1).y); \
        (c1).z = fmaf((a1).z, (b), (c1).z); \
        (c1).w = fmaf((a1).w, (b), (c1).w); \
    } while (0)

__global__ void matmul(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C) {
    __shared__ __align__(16) float subtileA[2][BLOCK_K][BLOCK_M];
    __shared__ __align__(16) float subtileB[2][BLOCK_K][BLOCK_N + 4];

    int linear_block = blockIdx.y * gridDim.x + blockIdx.x;
    int bx = linear_block / gridDim.y;
    int by = linear_block % gridDim.y;
    int tid = threadIdx.x;

    int warp_id = tid / WARP_THREADS;
    int lane_id = tid % WARP_THREADS;
    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;
    int lane_m = lane_id % LANES_M;
    int lane_n = lane_id / LANES_M;

    float4 c0[THREAD_TILE_N] = {};
    float4 c1[THREAD_TILE_N] = {};

    int local_row_base = warp_m * WARP_TILE_M + lane_m * THREAD_TILE_M;
    int local_col_base = warp_n * WARP_TILE_N + lane_n * THREAD_TILE_N;
    int row_base = BLOCK_M * by + local_row_base;
    int col_base = BLOCK_N * bx + local_col_base;

    float4 pref_b[PREFETCH_VECS_PER_THREAD];

    for (int load_i = 0; load_i < PREFETCH_VECS_PER_THREAD; ++load_i) {
        int a_idx = tid + load_i * BLOCK_THREADS;
        int a_load_row = (a_idx % (BLOCK_M / FLOAT4_WIDTH)) * FLOAT4_WIDTH;
        int a_load_k = a_idx / (BLOCK_M / FLOAT4_WIDTH);
        int b_idx = tid + load_i * BLOCK_THREADS;
        int b_load_k = (b_idx % (BLOCK_K / FLOAT4_WIDTH)) * FLOAT4_WIDTH;
        int b_load_col = b_idx / (BLOCK_K / FLOAT4_WIDTH);

        pref_b[load_i] = reinterpret_cast<const float4 *>(
            &B[(BLOCK_N * bx + b_load_col) * MATRIX_K + b_load_k])[0];

        cp_async_float4(
            &subtileA[0][a_load_k][a_load_row],
            &A[a_load_k * MATRIX_M + BLOCK_M * by + a_load_row]);
        subtileB[0][b_load_k + 0][b_load_col] = pref_b[load_i].x;
        subtileB[0][b_load_k + 1][b_load_col] = pref_b[load_i].y;
        subtileB[0][b_load_k + 2][b_load_col] = pref_b[load_i].z;
        subtileB[0][b_load_k + 3][b_load_col] = pref_b[load_i].w;
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    for (int t = 0, read_stage = 0; t < MATRIX_K; t += BLOCK_K, read_stage ^= 1) {
        int next_t = t + BLOCK_K;
        bool has_next = next_t < MATRIX_K;
        int write_stage = read_stage ^ 1;

        if (has_next) {
            for (int load_i = 0; load_i < PREFETCH_VECS_PER_THREAD; ++load_i) {
                int a_idx = tid + load_i * BLOCK_THREADS;
                int a_load_row = (a_idx % (BLOCK_M / FLOAT4_WIDTH)) * FLOAT4_WIDTH;
                int a_load_k = a_idx / (BLOCK_M / FLOAT4_WIDTH);
                int b_idx = tid + load_i * BLOCK_THREADS;
                int b_load_k = (b_idx % (BLOCK_K / FLOAT4_WIDTH)) * FLOAT4_WIDTH;
                int b_load_col = b_idx / (BLOCK_K / FLOAT4_WIDTH);

                cp_async_float4(
                    &subtileA[write_stage][a_load_k][a_load_row],
                    &A[(next_t + a_load_k) * MATRIX_M + BLOCK_M * by + a_load_row]);
                pref_b[load_i] = reinterpret_cast<const float4 *>(
                    &B[(BLOCK_N * bx + b_load_col) * MATRIX_K + next_t + b_load_k])[0];
            }
            cp_async_commit();
        }

        for (int k = 0; k < BLOCK_K; ++k) {
            float4 a0 = reinterpret_cast<const float4 *>(&subtileA[read_stage][k][local_row_base + 0])[0];
            float4 a1 = reinterpret_cast<const float4 *>(&subtileA[read_stage][k][local_row_base + 4])[0];
            float4 b0 = reinterpret_cast<const float4 *>(&subtileB[read_stage][k][local_col_base + 0])[0];
            float4 b1 = reinterpret_cast<const float4 *>(&subtileB[read_stage][k][local_col_base + 4])[0];

            FMA_ACCUM_COL(a0, a1, b0.x, c0[0], c1[0]);
            FMA_ACCUM_COL(a0, a1, b0.y, c0[1], c1[1]);
            FMA_ACCUM_COL(a0, a1, b0.z, c0[2], c1[2]);
            FMA_ACCUM_COL(a0, a1, b0.w, c0[3], c1[3]);
            FMA_ACCUM_COL(a0, a1, b1.x, c0[4], c1[4]);
            FMA_ACCUM_COL(a0, a1, b1.y, c0[5], c1[5]);
            FMA_ACCUM_COL(a0, a1, b1.z, c0[6], c1[6]);
            FMA_ACCUM_COL(a0, a1, b1.w, c0[7], c1[7]);
        }

        if (has_next) {
            #pragma unroll
            for (int load_i = 0; load_i < PREFETCH_VECS_PER_THREAD; ++load_i) {
                int b_idx = tid + load_i * BLOCK_THREADS;
                int b_load_k = (b_idx % (BLOCK_K / FLOAT4_WIDTH)) * FLOAT4_WIDTH;
                int b_load_col = b_idx / (BLOCK_K / FLOAT4_WIDTH);

                subtileB[write_stage][b_load_k + 0][b_load_col] = pref_b[load_i].x;
                subtileB[write_stage][b_load_k + 1][b_load_col] = pref_b[load_i].y;
                subtileB[write_stage][b_load_k + 2][b_load_col] = pref_b[load_i].z;
                subtileB[write_stage][b_load_k + 3][b_load_col] = pref_b[load_i].w;
            }
            cp_async_wait_all();
            __syncthreads();
        }
    }

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_N; ++i) {
        int col = col_base + i;
        int r_idx = col * MATRIX_M + row_base;
        reinterpret_cast<float4 *>(&C[r_idx])[0] = c0[i];
        reinterpret_cast<float4 *>(&C[r_idx + FLOAT4_WIDTH])[0] = c1[i];
    }
}

#undef FMA_ACCUM_COL

void fill_random(float* arr, size_t size) {
    std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        arr[i] = dist(gen);
    }
}

bool validate(const float* A, const float* B, size_t size, float rtol = 1e-3f, float atol = 1e-2f) {
    for (size_t i = 0; i < size; ++i) {
        float diff = fabs(A[i] - B[i]);
        float tol = atol + rtol * fabs(A[i]);
        if (diff > tol) return false;
    }
    return true;
}

#ifdef USE_CUBLAS
const char* cublas_status_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}
#endif

double gflops(const VersionConfig& config, float elapsed_ms) {
    if (elapsed_ms <= 0.0f) return 0.0;
    double ops = 2.0 * (double)config.M * (double)config.N * (double)config.K;
    return ops / ((double)elapsed_ms * 1.0e6);
}

void print_time_result(const char* label, float elapsed_ms, const VersionConfig& config) {
    std::cout << ">>> " << label << " execution time: " << elapsed_ms
              << " ms (" << gflops(config, elapsed_ms) << " GFLOP/s)" << std::endl;
}

void print_time_comparison(float custom_ms, float cublas_ms) {
    std::cout << ">>> Custom kernel is " << (cublas_ms / custom_ms)
              << "x faster than cuBLAS SGEMM." << std::endl;
}

void profiler_range_push(const char* name) {
#ifdef USE_NVTX
    nvtxRangePushA(name);
#else
    (void)name;
#endif
}

void profiler_range_pop() {
#ifdef USE_NVTX
    nvtxRangePop();
#endif
}

void print_first_10(const float* arr, size_t size) {
    size_t limit = std::min(size, (size_t)10);
    for (size_t i = 0; i < limit; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
}

float* cpu(const float* A, const float* B, int A_height, int A_width, int B_width) {
    float* C = new float[(size_t)A_height * B_width]();

    omp_set_num_threads(6);
    #pragma omp parallel for
    for (int j = 0; j < B_width; ++j) {
        for (int k = 0; k < A_width; ++k) {
            float Bkj = B[k + j * A_width];
            for (int i = 0; i < A_height; ++i) {
                C[i + j * A_height] += A[i + k * A_height] * Bkj;
            }
        }
    }

    return C;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [--verify|-v] [--cublas|-b] [--help|-h]\n"
              << "  --verify, -v  Copy result back and run CPU validation.\n"
              << "  --cublas, -b  Compare custom kernel time with cuBLAS SGEMM.\n"
              << "  --help, -h    Show this help message.\n";
}

int main(int argc, char* argv[]) {

    bool verify = false;
    bool compare_cublas = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verify" || arg == "-v") {
            verify = true;
        } else if (arg == "--cublas" || arg == "-b") {
            compare_cublas = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

#ifndef USE_CUBLAS
    if (compare_cublas) {
        std::cerr << "cuBLAS comparison requested, but this binary was built without USE_CUBLAS.\n"
                  << "Rebuild with -DUSE_CUBLAS and link with -lcublas.\n";
        return 1;
    }
#endif

    VersionConfig config = {MATRIX_M, MATRIX_K, MATRIX_N};
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "----------------Mat Mul----------------\n";
    std::cout << "Verification: " << (verify ? "on" : "off") << "\n";
    std::cout << "cuBLAS comparison: " << (compare_cublas ? "on" : "off") << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERATIONS << "\n";
    std::cout << "Benchmark iterations: " << BENCHMARK_ITERATIONS << "\n";

    int A_height = config.M;
    int A_width = config.K;
    int B_height = config.K;
    int B_width = config.N;

    size_t A_size = (size_t)A_height * A_width;
    size_t B_size = (size_t)B_height * B_width;
    size_t C_size = (size_t)A_height * B_width;

    float* A_host = new float[A_size];
    float* B_host = new float[B_size];
    float* C_host = verify ? new float[C_size] : nullptr;
    float* C_cublas_host = (verify && compare_cublas) ? new float[C_size] : nullptr;

    fill_random(A_host, A_size);
    fill_random(B_host, B_size);

    float *A_dev, *B_dev, *C_dev, *C_cublas_dev = nullptr;
    cudaMalloc((void **)&A_dev, A_size * sizeof(float));
    cudaMalloc((void **)&B_dev, B_size * sizeof(float));
    cudaMalloc((void **)&C_dev, C_size * sizeof(float));
    if (compare_cublas) {
        cudaMalloc((void **)&C_cublas_dev, C_size * sizeof(float));
    }
    cudaMemcpy(A_dev, A_host, A_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, B_size * sizeof(float), cudaMemcpyHostToDevice);
    

    cudaEvent_t start, stop;
    float execution_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 dimGrid(MATRIX_N / BLOCK_N, MATRIX_M / BLOCK_M, 1);
    dim3 dimBlock(BLOCK_THREADS, 1, 1);
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        matmul<<<dimGrid, dimBlock>>>(A_dev, B_dev, C_dev);
    }
    cudaDeviceSynchronize();

    profiler_range_push("custom_matmul");
    cudaEventRecord(start); 
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        matmul<<<dimGrid, dimBlock>>>(A_dev, B_dev, C_dev);
    }
    cudaEventRecord(stop);

    cudaError_t launchErr = cudaGetLastError();
    cudaError_t syncErr = cudaEventSynchronize(stop);
    profiler_range_pop();
    if (launchErr == cudaSuccess && syncErr == cudaSuccess) {
        cudaEventElapsedTime(&execution_time, start, stop);
        execution_time /= BENCHMARK_ITERATIONS;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    if (launchErr != cudaSuccess) {
        std::cout << "  [CUDA ERROR]: " << cudaGetErrorString(launchErr) << std::endl;
    } else if (syncErr != cudaSuccess) {
        std::cout << "  [CUDA ERROR]: " << cudaGetErrorString(syncErr) << std::endl;
    } else {
        print_time_result("Custom kernel", execution_time, config);
    }

    bool cublasOk = false;
#ifdef USE_CUBLAS
    if (compare_cublas && launchErr == cudaSuccess && syncErr == cudaSuccess) {
        cublasHandle_t handle;
        cublasStatus_t cublasErr = cublasCreate(&handle);
        cudaError_t cublasSyncErr = cudaSuccess;
        float cublas_execution_time = 0;

        if (cublasErr == CUBLAS_STATUS_SUCCESS) {
            const float alpha = 1.0f;
            const float beta = 0.0f;

            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

            for (int i = 0; i < WARMUP_ITERATIONS && cublasErr == CUBLAS_STATUS_SUCCESS; ++i) {
                cublasErr = cublasSgemm(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    A_height, B_width, A_width,
                    &alpha,
                    A_dev, A_height,
                    B_dev, A_width,
                    &beta,
                    C_cublas_dev, A_height);
            }
            cublasSyncErr = cudaDeviceSynchronize();

            cudaEvent_t cublas_start, cublas_stop;
            cudaEventCreate(&cublas_start);
            cudaEventCreate(&cublas_stop);

            if (cublasErr == CUBLAS_STATUS_SUCCESS && cublasSyncErr == cudaSuccess) {
                profiler_range_push("cublas_sgemm");
                cudaEventRecord(cublas_start);
                for (int i = 0; i < BENCHMARK_ITERATIONS && cublasErr == CUBLAS_STATUS_SUCCESS; ++i) {
                    cublasErr = cublasSgemm(
                        handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        A_height, B_width, A_width,
                        &alpha,
                        A_dev, A_height,
                        B_dev, A_width,
                        &beta,
                        C_cublas_dev, A_height);
                }
                cudaEventRecord(cublas_stop);

                cublasSyncErr = cudaEventSynchronize(cublas_stop);
                profiler_range_pop();
            }
            if (cublasErr == CUBLAS_STATUS_SUCCESS && cublasSyncErr == cudaSuccess) {
                cudaEventElapsedTime(&cublas_execution_time, cublas_start, cublas_stop);
                cublas_execution_time /= BENCHMARK_ITERATIONS;
                cublasOk = true;
            }

            cudaEventDestroy(cublas_start);
            cudaEventDestroy(cublas_stop);
            cublasDestroy(handle);
        }

        if (cublasErr != CUBLAS_STATUS_SUCCESS) {
            std::cout << "  [cuBLAS ERROR]: " << cublas_status_string(cublasErr) << std::endl;
        } else if (cublasSyncErr != cudaSuccess) {
            std::cout << "  [cuBLAS CUDA ERROR]: " << cudaGetErrorString(cublasSyncErr) << std::endl;
        } else {
            print_time_result("cuBLAS SGEMM", cublas_execution_time, config);
            print_time_comparison(execution_time, cublas_execution_time);
        }
    } else if (compare_cublas) {
        std::cout << ">>> cuBLAS comparison skipped because the custom kernel did not complete successfully." << std::endl;
    }
#endif
    

    if (verify && launchErr == cudaSuccess && syncErr == cudaSuccess) {
        cudaMemcpy(C_host, C_dev, C_size * sizeof(float), cudaMemcpyDeviceToHost);

        float* C_answer = cpu(A_host, B_host, A_height, A_width, B_width);

        if (validate(C_answer, C_host, C_size, 1e-3f, 1e-2f)) {
            std::cout << ">>> Custom kernel test pass!" << std::endl;
        } else {
            std::cout << ">>> Custom kernel test fail!" << std::endl;
            std::cout << ">>> >>First 10 elements of C_answer: \n";
            print_first_10(C_answer, C_size);
            std::cout << ">>> >>First 10 elements of C_host: \n";
            print_first_10(C_host, C_size);
        }

        if (compare_cublas) {
            if (cublasOk) {
                cudaMemcpy(C_cublas_host, C_cublas_dev, C_size * sizeof(float), cudaMemcpyDeviceToHost);

                if (validate(C_answer, C_cublas_host, C_size, 1e-3f, 1e-2f)) {
                    std::cout << ">>> cuBLAS SGEMM test pass!" << std::endl;
                } else {
                    std::cout << ">>> cuBLAS SGEMM test fail!" << std::endl;
                    std::cout << ">>> >>First 10 elements of C_answer: \n";
                    print_first_10(C_answer, C_size);
                    std::cout << ">>> >>First 10 elements of C_cublas_host: \n";
                    print_first_10(C_cublas_host, C_size);
                }
            } else {
                std::cout << ">>> cuBLAS verification skipped because SGEMM did not complete successfully." << std::endl;
            }
        }

        delete[] C_answer;
    } else if (verify) {
        std::cout << ">>> Verification skipped because the kernel did not complete successfully." << std::endl;
    } else {
        std::cout << ">>> Verification skipped. Use --verify to enable it." << std::endl;
    }

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    if (C_cublas_dev != nullptr) {
        cudaFree(C_cublas_dev);
    }

    delete[] A_host;
    delete[] B_host;
    delete[] C_host;
    delete[] C_cublas_host;

    bool customOk = (launchErr == cudaSuccess && syncErr == cudaSuccess);
    bool allOk = customOk && (!compare_cublas || cublasOk);
    return allOk ? 0 : 1;
}
