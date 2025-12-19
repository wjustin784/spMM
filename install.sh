#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="spmm"
PY_VER="3.10"

# Resolve repo root as the directory where this script lives
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUPY_SRC="${REPO_ROOT}/modify_src/cupy-src"
REQ_FILE="${REPO_ROOT}/requirements.txt"

CUDA_PATH="/usr/local/cuda"

echo "[1/6] Checking conda..."
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found in PATH."; exit 1; }

echo "[2/6] Ensuring conda env exists: ${ENV_NAME} (python=${PY_VER})"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "  - Env '${ENV_NAME}' already exists. Skipping create."
else
  conda create -n "${ENV_NAME}" "python=${PY_VER}" -y
fi

echo "[3/6] Installing conda dependencies into env '${ENV_NAME}'"
conda install -n "${ENV_NAME}" -c conda-forge -y "libstdcxx-ng>=13.2.0" "libgcc-ng>=13.2.0"
conda install -n "${ENV_NAME}" -c conda-forge -y _libgcc_mutex sysroot_linux-64

echo "[4/6] Installing modified CuPy from source (editable) ..."
if [[ ! -d "${CUPY_SRC}" ]]; then
  echo "ERROR: CuPy source directory not found at: ${CUPY_SRC}"
  exit 1
fi

# Build/install CuPy with required env vars set for the install step
(
  cd "${CUPY_SRC}"
  env \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    LD=/usr/bin/ld \
    CUPY_NVCC_GENERATE_CODE=current \
    CUDA_PATH="${CUDA_PATH}" \
    PATH="${PATH}:${CUDA_PATH}/bin" \
    LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
    conda run -n "${ENV_NAME}" python -m pip install -e . -v
)

echo "[5/6] Installing Python requirements into env '${ENV_NAME}'"
if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR: requirements.txt not found at: ${REQ_FILE}"
  exit 1
fi
conda run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}"

echo "[6/6] Done."
echo "To use this environment in your current shell:"
echo "  conda activate ${ENV_NAME}"
