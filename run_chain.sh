#!/bin/bash
################################################################################
# Complete AI Processing Chain
# 1. Person Detection (person.py)
# 2. Face Clustering (cam2_cluster_faces.py)
################################################################################

# Script Directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
echo "│ STEP 1/2: Person Detection                                               │" | tee -a "${CHAIN_LOG}"
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
echo "│ STEP 2/2: Face Clustering                                                │" | tee -a "${CHAIN_LOG}"
echo "└──────────────────────────────────────────────────────────────────────────┘" | tee -a "${CHAIN_LOG}"
echo "" | tee -a "${CHAIN_LOG}"

"${SCRIPT_DIR}/run_cluster.sh" 2>&1 | tee -a "${CHAIN_LOG}"
CLUSTER_EXIT=$?

echo "" | tee -a "${CHAIN_LOG}"
if [ ${CLUSTER_EXIT} -ne 0 ]; then
    echo "⚠️  Face Clustering fehlgeschlagen (Exit Code: ${CLUSTER_EXIT})" | tee -a "${CHAIN_LOG}"
    echo "   Chain wird trotzdem als erfolgreich markiert" | tee -a "${CHAIN_LOG}"
else
    echo "✅ Face Clustering erfolgreich" | tee -a "${CHAIN_LOG}"
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
echo "  Step 1 (Person Detection): $([ ${PERSON_EXIT} -eq 0 ] && echo '✅ OK' || echo '❌ FEHLER')" | tee -a "${CHAIN_LOG}"
echo "  Step 2 (Face Clustering):  $([ ${CLUSTER_EXIT} -eq 0 ] && echo '✅ OK' || echo '⚠️  FEHLER')" | tee -a "${CHAIN_LOG}"
echo "" | tee -a "${CHAIN_LOG}"
echo "Komplettes Log: ${CHAIN_LOG}" | tee -a "${CHAIN_LOG}"
echo "##############################################################################" | tee -a "${CHAIN_LOG}"

# Exit mit Status von Person Detection (wichtigster Schritt)
exit ${PERSON_EXIT}
