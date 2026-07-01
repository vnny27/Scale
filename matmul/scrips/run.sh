#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${ROOT_DIR}/../src"
BUILD_DIR="${ROOT_DIR}/../build/benchmarks"
RESULTS_DIR="${ROOT_DIR}/../results"

NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
ARCH="${ARCH:-sm_89}"
COMMON_FLAGS=(-O3 -arch="${ARCH}" -DUSE_CUBLAS -Xcompiler -fopenmp -lcublas)
RUN_ARGS=(--cublas)

if [[ -n "${EXTRA_NVCC_FLAGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_FLAGS=(${EXTRA_NVCC_FLAGS})
else
  EXTRA_FLAGS=()
fi

for arg in "$@"; do
  case "${arg}" in
    --verify|-v)
      RUN_ARGS+=(--verify)
      ;;
    --no-cublas)
      COMMON_FLAGS=(-O3 -arch="${ARCH}" -Xcompiler -fopenmp)
      RUN_ARGS=()
      ;;
    --clean)
      rm -rf "${BUILD_DIR}"
      ;;
    --help|-h)
      echo "Usage: ./run.sh [--verify] [--no-cublas] [--clean]"
      echo "Environment: NVCC=/path/to/nvcc ARCH=sm_89 EXTRA_NVCC_FLAGS='...'"
      exit 0
      ;;
    *)
      echo "Unknown option: ${arg}" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"

SOURCES=(
  "v00_base_matmul.cu"
  "v01_tiled_matmul.cu"
  "v02_micro4_matmul.cu"
  "v03_interleaved_b_matmul.cu"
  "v04_b_repack_matmul.cu"
  "v05_warp_tiled_matmul.cu"
  "v06_vectorized_matmul.cu"
  "v07_factorized_compute.cu"
  "v08_lane_mapping.cu"
  "v09_double_buffering.cu"
  "v10_cp_async.cu"
)

STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY="${RESULTS_DIR}/benchmark_${STAMP}.tsv"
LOG="${RESULTS_DIR}/benchmark_${STAMP}.log"
printf "version\tstatus\tcustom_ms\tcustom_gflops\tcublas_ms\tcublas_gflops\tcublas_percent\n" > "${SUMMARY}"
printf "CUDA core benchmark log\n" > "${LOG}"

echo "CUDA core benchmark"
echo "  NVCC: ${NVCC}"
echo "  ARCH: ${ARCH}"
echo "  Summary: ${SUMMARY}"
echo "  Log: ${LOG}"
echo

for src in "${SOURCES[@]}"; do
  version="${src%%_*}"
  exe="${BUILD_DIR}/${src%.cu}"

  echo "==> Building ${src}"
  {
    echo
    echo "===== ${src} / build ====="
    echo "${NVCC} ${COMMON_FLAGS[*]} ${EXTRA_FLAGS[*]} ${SRC_DIR}/${src} -o ${exe}"
  } >> "${LOG}"
  if ! "${NVCC}" "${COMMON_FLAGS[@]}" "${EXTRA_FLAGS[@]}" "${SRC_DIR}/${src}" -o "${exe}" >> "${LOG}" 2>&1; then
    echo "    build failed; see ${LOG}"
    printf "%s\tbuild_failed\t\t\t\t\t\n" "${version}" >> "${SUMMARY}"
    continue
  fi

  echo "==> Running ${src}"
  {
    echo
    echo "===== ${src} / run ====="
    echo "${exe} ${RUN_ARGS[*]}"
  } >> "${LOG}"
  if "${exe}" "${RUN_ARGS[@]}" >> "${LOG}" 2>&1; then
    status="ok"
  else
    status="run_failed"
    echo "    run failed; see ${LOG}"
  fi

  parsed="$(awk -v marker="===== ${src} / run =====" '
    $0 == marker { in_block=1; next }
    /^===== .* \/ (build|run) =====/ && in_block { in_block=0 }
    in_block && />>> Custom kernel execution time:/ {
      custom_ms=$6; custom_gflops=$8; gsub(/[()]/, "", custom_gflops)
    }
    in_block && />>> cuBLAS SGEMM execution time:/ {
      cublas_ms=$6; cublas_gflops=$8; gsub(/[()]/, "", cublas_gflops)
    }
    END {
      if (custom_gflops != "" && cublas_gflops != "" && cublas_gflops > 0) {
        cublas_percent = custom_gflops / cublas_gflops * 100.0
        printf "%s\t%s\t%s\t%s\t%.1f", custom_ms, custom_gflops, cublas_ms, cublas_gflops, cublas_percent
      } else {
        printf "%s\t%s\t%s\t%s\t", custom_ms, custom_gflops, cublas_ms, cublas_gflops
      }
    }
  ' "${LOG}")"

  IFS=$'\t' read -r custom_ms custom_gflops cublas_ms cublas_gflops cublas_percent <<< "${parsed}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${version}" "${status}" "${custom_ms}" "${custom_gflops}" \
    "${cublas_ms}" "${cublas_gflops}" "${cublas_percent}" >> "${SUMMARY}"
done

echo
echo "Done. Summary:"
column -t -s $'\t' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
