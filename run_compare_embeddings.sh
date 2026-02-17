#!/bin/bash
################################################################################
# Embedding Comparison Wrapper Script
################################################################################

VENV_BIN="/home/gh/python/venv_py311/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_SCRIPT="${SCRIPT_DIR}/compare_all_embeddings.py"

echo "=============================================================================="
echo "Embedding Vergleich gestartet: $(date)"
echo "=============================================================================="
echo "Python: ${VENV_BIN}"
echo "Script: ${COMPARE_SCRIPT}"
echo ""

"${VENV_BIN}" "${COMPARE_SCRIPT}" "$@"

EXIT_CODE=$?

echo ""
echo "=============================================================================="
echo "Embedding Vergleich beendet: $(date) (Exit Code: ${EXIT_CODE})"
echo "=============================================================================="

exit ${EXIT_CODE}
