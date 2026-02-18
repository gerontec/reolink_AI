#!/bin/bash
################################################################################
# Complete AI Processing Chain
# 1. Person Detection (person.py)
# 2. Face Clustering (cam2_cluster_faces.py)
# 3. Statistics Report (cam2_report.py)
################################################################################

# Script Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# LOCK-FILE Mechanismus (verhindert parallele Ausführung)
# ============================================================================
LOCK_FILE="${SCRIPT_DIR}/.chain.lock"
PID_FILE="${SCRIPT_DIR}/.chain.pid"

# Prüfe ob bereits eine Instanz läuft
if [ -f "${LOCK_FILE}" ]; then
    # Prüfe ob der Prozess noch existiert
    if [ -f "${PID_FILE}" ]; then
        OLD_PID=$(cat "${PID_FILE}")
        if kill -0 "${OLD_PID}" 2>/dev/null; then
            echo "❌ Chain läuft bereits (PID: ${OLD_PID})"
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

# Log-Datei für die gesamte Chain (wird überschrieben)
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
CHAIN_LOG="${LOG_DIR}/chain.log"

# Alte Log als Backup behalten
if [ -f "${CHAIN_LOG}" ]; then
    mv "${CHAIN_LOG}" "${CHAIN_LOG}.old"
fi

echo "##############################################################################" | tee "${CHAIN_LOG}"
echo "#  AI Processing Chain gestartet: $(date)" | tee -a "${CHAIN_LOG}"
echo "##############################################################################" | tee -a "${CHAIN_LOG}"
echo "" | tee -a "${CHAIN_LOG}"

# Gesamtzeit messen
START_TIME=$(date +%s)

# ============================================================================
# STEP 1: Person Detection
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐" | tee -a "${CHAIN_LOG}"
echo "│ STEP 1/3: Person Detection                                               │" | tee -a "${CHAIN_LOG}"
echo "└──────────────────────────────────────────────────────────────────────────┘" | tee -a "${CHAIN_LOG}"
echo "" | tee -a "${CHAIN_LOG}"

"${SCRIPT_DIR}/run_person.sh" "$@" 2>&1 | tee -a "${CHAIN_LOG}"
PERSON_EXIT=$?

echo "" | tee -a "${CHAIN_LOG}"
if [ ${PERSON_EXIT} -ne 0 ]; then
    echo "❌ Person Detection fehlgeschlagen (Exit Code: ${PERSON_EXIT})" | tee -a "${CHAIN_LOG}"
    echo "   Chain wird ABGEBROCHEN" | tee -a "${CHAIN_LOG}"
    exit ${PERSON_EXIT}
else
    echo "✅ Person Detection erfolgreich" | tee -a "${CHAIN_LOG}"
fi
echo "" | tee -a "${CHAIN_LOG}"

# ============================================================================
# STEP 2: Face Clustering (nur wenn neue Gesichter erkannt wurden)
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐" | tee -a "${CHAIN_LOG}"
echo "│ STEP 2/3: Face Clustering                                                │" | tee -a "${CHAIN_LOG}"
echo "└──────────────────────────────────────────────────────────────────────────┘" | tee -a "${CHAIN_LOG}"
echo "" | tee -a "${CHAIN_LOG}"

"${SCRIPT_DIR}/run_cluster.sh" 2>&1 | tee -a "${CHAIN_LOG}"
CLUSTER_EXIT=$?

echo "" | tee -a "${CHAIN_LOG}"
if [ ${CLUSTER_EXIT} -ne 0 ]; then
    echo "⚠️  Face Clustering fehlgeschlagen (Exit Code: ${CLUSTER_EXIT})" | tee -a "${CHAIN_LOG}"
    echo "   Chain wird trotzdem fortgesetzt" | tee -a "${CHAIN_LOG}"
else
    echo "✅ Face Clustering erfolgreich" | tee -a "${CHAIN_LOG}"
fi
echo "" | tee -a "${CHAIN_LOG}"

# ============================================================================
# STEP 3: Statistics Report
# ============================================================================
echo "┌──────────────────────────────────────────────────────────────────────────┐" | tee -a "${CHAIN_LOG}"
echo "│ STEP 3/3: Statistics Report                                              │" | tee -a "${CHAIN_LOG}"
echo "└──────────────────────────────────────────────────────────────────────────┘" | tee -a "${CHAIN_LOG}"
echo "" | tee -a "${CHAIN_LOG}"

"${SCRIPT_DIR}/run_report.sh" 2>&1 | tee -a "${CHAIN_LOG}"
REPORT_EXIT=$?

echo "" | tee -a "${CHAIN_LOG}"
if [ ${REPORT_EXIT} -ne 0 ]; then
    echo "⚠️  Statistics Report fehlgeschlagen (Exit Code: ${REPORT_EXIT})" | tee -a "${CHAIN_LOG}"
    echo "   Chain wird trotzdem als erfolgreich markiert" | tee -a "${CHAIN_LOG}"
else
    echo "✅ Statistics Report erfolgreich" | tee -a "${CHAIN_LOG}"
fi

# ============================================================================
# Summary
# ============================================================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo "" | tee -a "${CHAIN_LOG}"
echo "##############################################################################" | tee -a "${CHAIN_LOG}"
echo "#  AI Processing Chain beendet: $(date)" | tee -a "${CHAIN_LOG}"
echo "##############################################################################" | tee -a "${CHAIN_LOG}"
echo "" | tee -a "${CHAIN_LOG}"
echo "Gesamt-Dauer: ${MINUTES}m ${SECONDS}s" | tee -a "${CHAIN_LOG}"
echo "" | tee -a "${CHAIN_LOG}"
echo "Status:" | tee -a "${CHAIN_LOG}"
echo "  Step 1 (Person Detection):  $([ ${PERSON_EXIT} -eq 0 ] && echo '✅ OK' || echo '❌ FEHLER')" | tee -a "${CHAIN_LOG}"
echo "  Step 2 (Face Clustering):   $([ ${CLUSTER_EXIT} -eq 0 ] && echo '✅ OK' || echo '⚠️  FEHLER')" | tee -a "${CHAIN_LOG}"
echo "  Step 3 (Statistics Report): $([ ${REPORT_EXIT} -eq 0 ] && echo '✅ OK' || echo '⚠️  FEHLER')" | tee -a "${CHAIN_LOG}"
echo "" | tee -a "${CHAIN_LOG}"
echo "Komplettes Log: ${CHAIN_LOG}" | tee -a "${CHAIN_LOG}"
echo "##############################################################################" | tee -a "${CHAIN_LOG}"

# Exit mit Status von Person Detection (wichtigster Schritt)
exit ${PERSON_EXIT}
