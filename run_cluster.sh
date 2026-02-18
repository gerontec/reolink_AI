#!/bin/bash
################################################################################
# Face Clustering Wrapper Script
# Führt cam2_cluster_faces.py aus
################################################################################

# Python Virtual Environment
VENV_BIN="/home/gh/python/venv_py311/bin/python3"

# Script Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_SCRIPT="${SCRIPT_DIR}/cam2_cluster_faces.py"

# Log-Datei (wird überschrieben)
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/cluster.log"

# Alte Log als Backup behalten
if [ -f "${LOG_FILE}" ]; then
    mv "${LOG_FILE}" "${LOG_FILE}.old"
fi

echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "Face Clustering gestartet: $(date)" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "Python: ${VENV_BIN}" | tee -a "${LOG_FILE}"
echo "Script: ${CLUSTER_SCRIPT}" | tee -a "${LOG_FILE}"
echo "Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"

# Face Clustering ausführen
"${VENV_BIN}" "${CLUSTER_SCRIPT}" "$@" 2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=$?

echo "" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"
echo "Face Clustering beendet: $(date) (Exit Code: ${EXIT_CODE})" | tee -a "${LOG_FILE}"
echo "==============================================================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
