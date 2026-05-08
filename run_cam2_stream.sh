#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export CUDA_MODULE_LOADING=LAZY
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_BIN="/home/gh/python/venv_py311/bin/python3"
PID_FILE="${SCRIPT_DIR}/.cam2_stream.pid"
LOG_FILE="${SCRIPT_DIR}/logs/cam2_stream.log"
RESTART_DELAY=10

mkdir -p "${SCRIPT_DIR}/logs"

# Verhindern, dass mehrere Instanzen laufen
if [ -f "${PID_FILE}" ]; then
    OLD_PID=$(cat "${PID_FILE}")
    if kill -0 "${OLD_PID}" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] cam2_stream läuft bereits (PID ${OLD_PID})" | tee -a "${LOG_FILE}"
        exit 1
    fi
fi

echo $$ > "${PID_FILE}"
trap "rm -f '${PID_FILE}'; exit" INT TERM EXIT

echo "[$(date '+%Y-%m-%d %H:%M:%S')] cam2_stream Daemon gestartet (PID $$)" | tee -a "${LOG_FILE}"

while true; do
    "${VENV_BIN}" "${SCRIPT_DIR}/cam2_streamOO.py" "$@"
    EXIT_CODE=$?

    if [ "${EXIT_CODE}" -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] cam2_stream sauber beendet (exit 0)" | tee -a "${LOG_FILE}"
        rm -f "${PID_FILE}"
        exit 0
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] cam2_stream abgestürzt (exit ${EXIT_CODE}) – Neustart in ${RESTART_DELAY}s" | tee -a "${LOG_FILE}"
    sleep "${RESTART_DELAY}"
done
