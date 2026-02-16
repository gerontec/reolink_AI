#!/bin/bash
# Watchdog2 Wrapper mit CUDA 11.8 Support

export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export CUDA_MODULE_LOADING=LAZY
export ORT_DISABLE_CUDNN_FRONTEND=1
export CUDA_VISIBLE_DEVICES=0

PYTHON_BIN="/home/gh/python/venv_py311/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG_SCRIPT="${SCRIPT_DIR}/watchdog2.py"

DEFAULT_OPTS="--save-annotated --debug --limit 5 --det-thresh 0.4"

exec "${PYTHON_BIN}" "${WATCHDOG_SCRIPT}" ${DEFAULT_OPTS} "$@"
