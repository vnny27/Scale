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
#define WARMUP_ITERATIONS 2
#define BENCHMARK_ITERATIONS 10


#define MATRIX_M 4096
#define MATRIX_N 4096
#define MATRIX_K 4096
#define MMA_M 16
#define MMA_N 16
#define MMA_K 8
#define WARPS_M 4
#define WARPS_N 2
#define WARPS_PER_BLOCK (WARPS_M * WARPS_N)
#define WARP_THREADS 32
#define BLOCK_THREADS (WARPS_PER_BLOCK * WARP_THREADS)
#define BLOCK_M (WARPS_M * MMA_M)
#define BLOCK_N (WARPS_N * MMA_N)
#define BLOCK_K 32

__device__ __forceinline__ unsigned int f32_to_tf32(float x) {
    unsigned int y;
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void mma_m16n8k8_tf32(
    float d[4], const unsigned int a[4], const unsigned int b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ int swizzle_a_col(int row, int col) {
    return col ^ ((row & 7) << 2);
}

__device__ __forceinline__ int swizzle_b_col(int row, int col) {
    return col ^ ((row & 3) << 3);
}
__global__ void matmul(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C) {
    __shared__ float shared_a[BLOCK_M][BLOCK_K + 1];
    __shared__ float shared_b[BLOCK_K][BLOCK_N + 1];

    int warp_id = threadIdx.x / WARP_THREADS;
    int lane_id = threadIdx.x % WARP_THREADS;
    int group_id = lane_id / 4;
    int thread_in_group = lane_id % 4;

    int warp_m = warp_id / WARPS_N;
    int warp_n = warp_id % WARPS_N;

    int blocks_n = MATRIX_N / BLOCK_N;
    int block_m = blockIdx.x / blocks_n;
    int block_n = blockIdx.x % blocks_n;

    int block_base_row = block_m * BLOCK_M;
    int block_base_col = block_n * BLOCK_N;
    int base_row = block_base_row + warp_m * MMA_M;
    int base_col = block_base_col + warp_n * MMA_N;

    float acc_left[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc_right[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k = 0; k < MATRIX_K; k += BLOCK_K) {
        for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_K; idx += BLOCK_THREADS) {
            int row = idx % BLOCK_M;
            int col = idx / BLOCK_M;
            shared_a[row][swizzle_a_col(row, col)] = A[(block_base_row + row) + (k + col) * MATRIX_M];
        }

        for (int idx = threadIdx.x; idx < BLOCK_K * BLOCK_N; idx += BLOCK_THREADS) {
            int row = idx % BLOCK_K;
            int col = idx / BLOCK_K;
            shared_b[row][swizzle_b_col(row, col)] = B[(k + row) + (block_base_col + col) * MATRIX_K];
        }

        __syncthreads();

        for (int kk = 0; kk < BLOCK_K; kk += MMA_K) {
            unsigned int a_frag[4];
            unsigned int b_left[2];
            unsigned int b_right[2];

            int warp_row_base = warp_m * MMA_M;
            int warp_col_base = warp_n * MMA_N;

            int a_row0 = warp_row_base + group_id;
            int a_row1 = warp_row_base + group_id + 8;
            int a_col0 = kk + thread_in_group;
            int a_col1 = kk + thread_in_group + 4;
            a_frag[0] = f32_to_tf32(shared_a[a_row0][swizzle_a_col(a_row0, a_col0)]);
            a_frag[1] = f32_to_tf32(shared_a[a_row1][swizzle_a_col(a_row1, a_col0)]);
            a_frag[2] = f32_to_tf32(shared_a[a_row0][swizzle_a_col(a_row0, a_col1)]);
            a_frag[3] = f32_to_tf32(shared_a[a_row1][swizzle_a_col(a_row1, a_col1)]);

            int b_col_left = warp_col_base + group_id;
            int b_col_right = warp_col_base + group_id + 8;
            int b_row0 = kk + thread_in_group;
            int b_row1 = kk + thread_in_group + 4;
            b_left[0] = f32_to_tf32(shared_b[b_row0][swizzle_b_col(b_row0, b_col_left)]);
            b_left[1] = f32_to_tf32(shared_b[b_row1][swizzle_b_col(b_row1, b_col_left)]);
            b_right[0] = f32_to_tf32(shared_b[b_row0][swizzle_b_col(b_row0, b_col_right)]);
            b_right[1] = f32_to_tf32(shared_b[b_row1][swizzle_b_col(b_row1, b_col_right)]);

            float out_left[4];
            float out_right[4];
            mma_m16n8k8_tf32(out_left, a_frag, b_left, acc_left);
            mma_m16n8k8_tf32(out_right, a_frag, b_right, acc_right);
            acc_left[0] = out_left[0];
            acc_left[1] = out_left[1];
            acc_left[2] = out_left[2];
            acc_left[3] = out_left[3];
            acc_right[0] = out_right[0];
            acc_right[1] = out_right[1];
            acc_right[2] = out_right[2];
            acc_right[3] = out_right[3];
        }

        __syncthreads();
    }

    int c_col0 = thread_in_group * 2;
    int c_row0 = group_id;
    int c_row1 = group_id + 8;
    C[(base_row + c_row0) + (base_col + c_col0 + 0) * MATRIX_M] = acc_left[0];
    C[(base_row + c_row0) + (base_col + c_col0 + 1) * MATRIX_M] = acc_left[1];
    C[(base_row + c_row1) + (base_col + c_col0 + 0) * MATRIX_M] = acc_left[2];
    C[(base_row + c_row1) + (base_col + c_col0 + 1) * MATRIX_M] = acc_left[3];

    int right_col = base_col + 8;
    C[(base_row + c_row0) + (right_col + c_col0 + 0) * MATRIX_M] = acc_right[0];
    C[(base_row + c_row0) + (right_col + c_col0 + 1) * MATRIX_M] = acc_right[1];
    C[(base_row + c_row1) + (right_col + c_col0 + 0) * MATRIX_M] = acc_right[2];
    C[(base_row + c_row1) + (right_col + c_col0 + 1) * MATRIX_M] = acc_right[3];
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

    int total_blocks = (MATRIX_M / BLOCK_M) * (MATRIX_N / BLOCK_N);
    dim3 dimGrid(total_blocks, 1, 1);
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

            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

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

        if (validate(C_answer, C_host, C_size, 1e-2f, 1e-1f)) {
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

                if (validate(C_answer, C_cublas_host, C_size, 1e-2f, 1e-1f)) {
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
