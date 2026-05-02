#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export CUDA_MODULE_LOADING=LAZY
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="/home/gh/python/venv_py311/bin/python3"
PID_FILE="${SCRIPT_DIR}/.cam2_stream.pid"

mkdir -p "${SCRIPT_DIR}/logs"

exec "${VENV_BIN}" "${SCRIPT_DIR}/cam2_stream.py" "$@"
