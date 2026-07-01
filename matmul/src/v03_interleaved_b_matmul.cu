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
#define BENCHMARK_ITERATIONS 1


#define TILE 64
#define THREAD_TILE 4
#define K_TILE (TILE / 2)
#define BLOCK_THREADS (TILE / THREAD_TILE)
__global__ void matmul(float *A, float *B, float *C,int M, int N, int K) {
    __shared__ float subtileA[TILE][K_TILE];
    __shared__ float subtileB[K_TILE][BLOCK_THREADS][THREAD_TILE + 2];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_base = TILE * by + ty * THREAD_TILE;
    int col_base = TILE * bx + tx;
    int linear_tid = ty * BLOCK_THREADS + tx;
    int block_threads = BLOCK_THREADS * BLOCK_THREADS;

    float c_val[THREAD_TILE][THREAD_TILE] = {0.0f};

    for (int t = 0; t < K; t += K_TILE){
        for (int idx = linear_tid; idx < TILE * K_TILE; idx += block_threads) {
            int row = idx / K_TILE;
            int col = idx % K_TILE;
            int global_row = TILE * by + row;
            int global_col = t + col;

            if (global_row < M && global_col < K)
                subtileA[row][col] = A[global_row * K + global_col];
            else
                subtileA[row][col] = 0.0f;
        }

        for (int idx = linear_tid; idx < K_TILE * TILE; idx += block_threads) {
            int row = idx / TILE;
            int col = idx % TILE;
            int global_row = t + row;
            int global_col = TILE * bx + col;
            int shared_tx = col % BLOCK_THREADS;
            int shared_j = col / BLOCK_THREADS;

            if (global_row < K && global_col < N)
                subtileB[row][shared_tx][shared_j] = B[global_row * N + global_col];
            else
                subtileB[row][shared_tx][shared_j] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < K_TILE; k++){
            float a_val[THREAD_TILE];
            float b_val[THREAD_TILE];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE; ++i) {
                a_val[i] = subtileA[ty * THREAD_TILE + i][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE; ++j) {
                b_val[j] = subtileB[k][tx][j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE; ++j) {
                    c_val[i][j] += a_val[i] * b_val[j];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_TILE; ++i) {
        int row = row_base + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; ++j) {
                int col = col_base + j * BLOCK_THREADS;
                if (col < N) {
                    C[row * N + col] = c_val[i][j];
                }
            }
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
    for (int i = 0; i < A_height; ++i) {
        for (int k = 0; k < A_width; ++k) {
            float Aik = A[i * A_width + k];
            for (int j = 0; j < B_width; ++j) {
                C[i * B_width + j] += Aik * B[k * B_width + j];
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

    VersionConfig config = {4096, 4096, 4096};
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "----------------Mat Mul----------------\n";
    std::cout << "Verification: " << (verify ? "on" : "off") << "\n";
    std::cout << "cuBLAS comparison: " << (compare_cublas ? "on" : "off") << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERATIONS << "\n";
    std::cout << "Benchmark iterations: " << BENCHMARK_ITERATIONS << "\n";

    // Random input matrix
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

    dim3 dimGrid((config.N + TILE - 1) / TILE, (config.M + TILE - 1) / TILE, 1);
    dim3 dimBlock(BLOCK_THREADS, BLOCK_THREADS, 1);
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        matmul<<<dimGrid, dimBlock>>>(A_dev, B_dev, C_dev, config.M, config.N, config.K);
    }
    cudaDeviceSynchronize();

    profiler_range_push("custom_matmul");
    cudaEventRecord(start); 
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        matmul<<<dimGrid, dimBlock>>>(A_dev, B_dev, C_dev, config.M, config.N, config.K);
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
                    B_width, A_height, A_width,
                    &alpha,
                    B_dev, B_width,
                    A_dev, A_width,
                    &beta,
                    C_cublas_dev, B_width);
            }
            cublasSyncErr = cudaDeviceSynchronize();

            cudaEvent_t cublas_start, cublas_stop;
            cudaEventCreate(&cublas_start);
            cudaEventCreate(&cublas_stop);

            if (cublasErr == CUBLAS_STATUS_SUCCESS && cublasSyncErr == cudaSuccess) {
                profiler_range_push("cublas_sgemm");
                cudaEventRecord(cublas_start);
                // Row-major C = A * B is column-major C^T = B^T * A^T.
                for (int i = 0; i < BENCHMARK_ITERATIONS && cublasErr == CUBLAS_STATUS_SUCCESS; ++i) {
                    cublasErr = cublasSgemm(
                        handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        B_width, A_height, A_width,
                        &alpha,
                        B_dev, B_width,
                        A_dev, A_width,
                        &beta,
                        C_cublas_dev, B_width);
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





