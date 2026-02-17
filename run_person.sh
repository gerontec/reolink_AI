#!/bin/bash
################################################################################
# Person Detection Wrapper Script
# Führt person.py mit CUDA 11.8 Support aus
################################################################################

# CUDA 11.8 Library Path für ONNX Runtime GPU Support
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# Python Virtual Environment
VENV_BIN="/home/gh/python/venv_py311/bin/python3"

# Script Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSON_SCRIPT="${SCRIPT_DIR}/person.py"

# Log-Datei (wird überschrieben)
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/person.log"

# Alte Log als Backup behalten
if [ -f "${LOG_FILE}" ]; then
    mv "${LOG_FILE}" "${LOG_FILE}.old"
fi

# Standard-Optionen (kann über Parameter überschrieben werden)
# --jpg-only: Nur JPG-Dateien (schneller, ~0.1s pro Datei)
# --debug: Debug-Logging
# --limit: Maximale Anzahl Dateien (z.B. --limit 100)
DEFAULT_OPTS="--jpg-only --debug"

echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "Person Detection gestartet: $(date)" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "Python: ${VENV_BIN}" | tee -a "${LOG_FILE}"
echo "Script: ${PERSON_SCRIPT}" | tee -a "${LOG_FILE}"
echo "Options: ${DEFAULT_OPTS} $@" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Person Detection ausführen
"${VENV_BIN}" "${PERSON_SCRIPT}" ${DEFAULT_OPTS} "$@" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "Person Detection beendet: $(date) (Exit Code: ${EXIT_CODE})" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
