#!/bin/bash
# Watchdog2 Wrapper Script mit CUDA 11.8 Support
# Setzt automatisch die benötigten Environment-Variablen

# CUDA 11.8 Library Path für ONNX Runtime GPU Support
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Python Virtual Environment Path
PYTHON_BIN="/home/gh/python/venv_py311/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCHDOG_SCRIPT="${SCRIPT_DIR}/watchdog2.py"

# Standard-Optionen: Immer annotierte Bilder speichern
DEFAULT_OPTS="--save-annotated"

# Führe watchdog2.py mit Standard-Optionen + allen übergebenen Parametern aus
exec "${PYTHON_BIN}" "${WATCHDOG_SCRIPT}" ${DEFAULT_OPTS} "$@"
