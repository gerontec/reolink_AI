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

# ============================================================================
# LOCK-FILE Mechanismus (verhindert parallele Ausführung)
# ============================================================================
LOCK_FILE="${SCRIPT_DIR}/.person.lock"
PID_FILE="${SCRIPT_DIR}/.person.pid"

# Prüfe ob bereits eine Instanz läuft
if [ -f "${LOCK_FILE}" ]; then
    # Prüfe ob der Prozess noch existiert
    if [ -f "${PID_FILE}" ]; then
        OLD_PID=$(cat "${PID_FILE}")
        if kill -0 "${OLD_PID}" 2>/dev/null; then
            echo "❌ Person Detection läuft bereits (PID: ${OLD_PID})"
            echo "   Lock-File: ${LOCK_FILE}"
            echo "   Falls der Prozess hängt: kill ${OLD_PID}"
            exit 1
        else
            echo "⚠️  Stale Lock-File gefunden (Prozess ${OLD_PID} existiert nicht mehr)"
            echo "   Entferne Lock-File und starte neu..."
            rm -f "${LOCK_FILE}" "${PID_FILE}"
        fi
    fi
fi

# Lock-File erstellen
echo $$ > "${PID_FILE}"
touch "${LOCK_FILE}"

# Cleanup-Funktion (wird bei Exit aufgerufen)
cleanup() {
    rm -f "${LOCK_FILE}" "${PID_FILE}"
}

# Cleanup bei Exit, Interrupt, Terminate
trap cleanup EXIT INT TERM

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
