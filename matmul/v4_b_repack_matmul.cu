#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#include <cuda_runtime_api.h>
#ifdef USE_CUBLAS
#include <cublas_v2.h>
#endif
#ifdef USE_NVTX
#include <nvToolsExt.h>
#endif

struct VersionConfig {
    int M;
    int K;
    int N;
};

#define SEED 1234
#define WARMUP_ITERATIONS 10
#define BENCHMARK_ITERATIONS 1

#define MATRIX_M 4096
#define MATRIX_N 4096
#define MATRIX_K 4096

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 32
#define THREAD_TILE_M 4
#define THREAD_TILE_N 4
#define THREADS_M (BLOCK_M / THREAD_TILE_M)
#define THREADS_N (BLOCK_N / THREAD_TILE_N)
#define BLOCK_THREADS (THREADS_M * THREADS_N)

__global__ void v4_b_repack_matmul(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) {
    __shared__ float sharedA[BLOCK_M][BLOCK_K];
    __shared__ float sharedB[BLOCK_K][BLOCK_N];

    int thread_m = threadIdx.y;
    int thread_n = threadIdx.x;
    int tid = thread_m * THREADS_N + thread_n;

    int local_row = thread_m * THREAD_TILE_M;
    int local_col = thread_n * THREAD_TILE_N;
    int global_row = blockIdx.y * BLOCK_M + local_row;
    int global_col = blockIdx.x * BLOCK_N + local_col;

    float c_val[THREAD_TILE_M][THREAD_TILE_N] = {};

    constexpr int A_LOADS_PER_THREAD = (BLOCK_M * BLOCK_K) / BLOCK_THREADS;
    constexpr int B_LOADS_PER_THREAD = (BLOCK_N * BLOCK_K) / BLOCK_THREADS;

    for (int tile_k = 0; tile_k < MATRIX_K; tile_k += BLOCK_K) {
        #pragma unroll
        for (int load_i = 0; load_i < A_LOADS_PER_THREAD; ++load_i) {
            int a_idx = tid + load_i * BLOCK_THREADS;
            int a_row = a_idx / BLOCK_K;
            int a_k = a_idx % BLOCK_K;

            sharedA[a_row][a_k] =
                A[(blockIdx.y * BLOCK_M + a_row) * MATRIX_K + tile_k + a_k];
        }

        #pragma unroll
        for (int load_i = 0; load_i < B_LOADS_PER_THREAD; ++load_i) {
            int b_idx = tid + load_i * BLOCK_THREADS;
            int b_k = b_idx / BLOCK_N;
            int b_col = b_idx % BLOCK_N;

            sharedB[b_k][b_col] =
                B[(tile_k + b_k) * MATRIX_N + blockIdx.x * BLOCK_N + b_col];
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            float a_val[THREAD_TILE_M];
            float b_val[THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                a_val[i] = sharedA[local_row + i][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; ++j) {
                b_val[j] = sharedB[k][local_col + j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; ++j) {
                    c_val[i][j] = fmaf(a_val[i], b_val[j], c_val[i][j]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            C[(global_row + i) * MATRIX_N + global_col + j] = c_val[i][j];
        }
    }
}

void fill_random(float* arr, size_t size) {
    std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        arr[i] = dist(gen);
    }
}

bool validate(const float* expected, const float* actual, size_t size,
              float rtol = 1e-3f, float atol = 1e-2f) {
    for (size_t i = 0; i < size; ++i) {
        float diff = std::fabs(expected[i] - actual[i]);
        float tol = atol + rtol * std::fabs(expected[i]);
        if (diff > tol) {
            return false;
        }
    }
    return true;
}

float* cpu_matmul(const float* A, const float* B, int M, int K, int N) {
    float* C = new float[(size_t)M * N]();

    for (int row = 0; row < M; ++row) {
        for (int k = 0; k < K; ++k) {
            float a = A[row * K + k];
            for (int col = 0; col < N; ++col) {
                C[row * N + col] += a * B[k * N + col];
            }
        }
    }

    return C;
}

double gflops(const VersionConfig& config, float elapsed_ms) {
    if (elapsed_ms <= 0.0f) {
        return 0.0;
    }
    double ops = 2.0 * (double)config.M * (double)config.N * (double)config.K;
    return ops / ((double)elapsed_ms * 1.0e6);
}

void print_time_result(const char* label, float elapsed_ms,
                       const VersionConfig& config) {
    std::cout << ">>> " << label << " execution time: " << elapsed_ms
              << " ms (" << gflops(config, elapsed_ms) << " GFLOP/s)"
              << std::endl;
}

void print_time_comparison(float custom_ms, float cublas_ms) {
    if (custom_ms <= 0.0f || cublas_ms <= 0.0f) return;

    std::cout << ">>> Custom time vs cuBLAS: "
              << (custom_ms / cublas_ms)
              << "x (cuBLAS = 1.000x)" << std::endl;
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

void print_first_10(const float* arr, size_t size) {
    size_t limit = std::min(size, (size_t)10);
    for (size_t i = 0; i < limit; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
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

    size_t A_size = (size_t)MATRIX_M * MATRIX_K;
    size_t B_size = (size_t)MATRIX_K * MATRIX_N;
    size_t C_size = (size_t)MATRIX_M * MATRIX_N;

    float* A_host = new float[A_size];
    float* B_host = new float[B_size];
    float* C_host = verify ? new float[C_size] : nullptr;
    float* C_cublas_host = (verify && compare_cublas) ? new float[C_size] : nullptr;

    fill_random(A_host, A_size);
    fill_random(B_host, B_size);

    float *A_dev = nullptr;
    float *B_dev = nullptr;
    float *C_dev = nullptr;
    float *C_cublas_dev = nullptr;

    cudaMalloc((void**)&A_dev, A_size * sizeof(float));
    cudaMalloc((void**)&B_dev, B_size * sizeof(float));
    cudaMalloc((void**)&C_dev, C_size * sizeof(float));
    if (compare_cublas) {
        cudaMalloc((void**)&C_cublas_dev, C_size * sizeof(float));
    }

    cudaMemcpy(A_dev, A_host, A_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, B_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(THREADS_N, THREADS_M, 1);
    dim3 dimGrid(MATRIX_N / BLOCK_N, MATRIX_M / BLOCK_M, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        v4_b_repack_matmul<<<dimGrid, dimBlock>>>(A_dev, B_dev, C_dev);
    }
    cudaDeviceSynchronize();

    profiler_range_push("custom_matmul");
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        v4_b_repack_matmul<<<dimGrid, dimBlock>>>(A_dev, B_dev, C_dev);
    }
    cudaEventRecord(stop);

    cudaError_t launchErr = cudaGetLastError();
    cudaError_t syncErr = cudaEventSynchronize(stop);
    profiler_range_pop();

    float execution_time = 0.0f;
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
    float cublas_execution_time = 0.0f;
    if (compare_cublas && launchErr == cudaSuccess && syncErr == cudaSuccess) {
        cublasHandle_t handle;
        cublasStatus_t cublasErr = cublasCreate(&handle);
        cudaError_t cublasSyncErr = cudaSuccess;

        if (cublasErr == CUBLAS_STATUS_SUCCESS) {
            const float alpha = 1.0f;
            const float beta = 0.0f;

            for (int i = 0; i < WARMUP_ITERATIONS &&
                            cublasErr == CUBLAS_STATUS_SUCCESS; ++i) {
                cublasErr = cublasSgemm(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    MATRIX_N, MATRIX_M, MATRIX_K,
                    &alpha,
                    B_dev, MATRIX_N,
                    A_dev, MATRIX_K,
                    &beta,
                    C_cublas_dev, MATRIX_N);
            }
            cublasSyncErr = cudaDeviceSynchronize();

            cudaEvent_t cublas_start, cublas_stop;
            cudaEventCreate(&cublas_start);
            cudaEventCreate(&cublas_stop);

            if (cublasErr == CUBLAS_STATUS_SUCCESS &&
                cublasSyncErr == cudaSuccess) {
                profiler_range_push("cublas_sgemm");
                cudaEventRecord(cublas_start);
                for (int i = 0; i < BENCHMARK_ITERATIONS &&
                                cublasErr == CUBLAS_STATUS_SUCCESS; ++i) {
                    cublasErr = cublasSgemm(
                        handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        MATRIX_N, MATRIX_M, MATRIX_K,
                        &alpha,
                        B_dev, MATRIX_N,
                        A_dev, MATRIX_K,
                        &beta,
                        C_cublas_dev, MATRIX_N);
                }
                cudaEventRecord(cublas_stop);
                cublasSyncErr = cudaEventSynchronize(cublas_stop);
                profiler_range_pop();
            }

            if (cublasErr == CUBLAS_STATUS_SUCCESS &&
                cublasSyncErr == cudaSuccess) {
                cudaEventElapsedTime(&cublas_execution_time,
                                     cublas_start, cublas_stop);
                cublas_execution_time /= BENCHMARK_ITERATIONS;
                cublasOk = true;
            }

            cudaEventDestroy(cublas_start);
            cudaEventDestroy(cublas_stop);
            cublasDestroy(handle);
        }

        if (cublasErr != CUBLAS_STATUS_SUCCESS) {
            std::cout << "  [cuBLAS ERROR]: " << cublas_status_string(cublasErr)
                      << std::endl;
        } else if (cublasSyncErr != cudaSuccess) {
            std::cout << "  [cuBLAS CUDA ERROR]: "
                      << cudaGetErrorString(cublasSyncErr) << std::endl;
        } else {
            print_time_result("cuBLAS SGEMM", cublas_execution_time, config);
            print_time_comparison(execution_time, cublas_execution_time);
        }
    }
#endif

    if (verify && launchErr == cudaSuccess && syncErr == cudaSuccess) {
        cudaMemcpy(C_host, C_dev, C_size * sizeof(float), cudaMemcpyDeviceToHost);
        float* C_answer = cpu_matmul(A_host, B_host, MATRIX_M, MATRIX_K, MATRIX_N);

        if (validate(C_answer, C_host, C_size)) {
            std::cout << ">>> Custom kernel test pass!" << std::endl;
        } else {
            std::cout << ">>> Custom kernel test fail!" << std::endl;
            std::cout << ">>> First 10 elements of C_answer:\n";
            print_first_10(C_answer, C_size);
            std::cout << ">>> First 10 elements of C_host:\n";
            print_first_10(C_host, C_size);
        }

        if (compare_cublas && cublasOk) {
            cudaMemcpy(C_cublas_host, C_cublas_dev,
                       C_size * sizeof(float), cudaMemcpyDeviceToHost);
            if (validate(C_answer, C_cublas_host, C_size)) {
                std::cout << ">>> cuBLAS SGEMM test pass!" << std::endl;
            } else {
                std::cout << ">>> cuBLAS SGEMM test fail!" << std::endl;
            }
        }

        delete[] C_answer;
    } else if (verify) {
        std::cout << ">>> Verification skipped because the kernel did not complete successfully."
                  << std::endl;
    } else {
        std::cout << ">>> Verification skipped. Use --verify to enable it."
                  << std::endl;
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





