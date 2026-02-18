#!/bin/bash
################################################################################
# Status & Management Script für AI Processing Chain
# Zeigt laufende Prozesse, Lock-Files, GPU-Auslastung
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==============================================================================="
echo "  AI Processing Chain - Status"
echo "==============================================================================="
echo ""

# ============================================================================
# Lock-Files prüfen
# ============================================================================
echo "🔒 Lock-Files:"
echo ""

check_lock() {
    local name=$1
    local lock_file="${SCRIPT_DIR}/.${name}.lock"
    local pid_file="${SCRIPT_DIR}/.${name}.pid"

    if [ -f "${lock_file}" ]; then
        if [ -f "${pid_file}" ]; then
            pid=$(cat "${pid_file}")
            if kill -0 "${pid}" 2>/dev/null; then
                echo "  ✅ ${name}: LÄUFT (PID: ${pid})"
                # Zeige wie lange schon
                ps -p "${pid}" -o etime= | xargs echo "     Laufzeit:"
            else
                echo "  ⚠️  ${name}: STALE LOCK (PID ${pid} existiert nicht mehr)"
                echo "     Lock-File: ${lock_file}"
                echo "     Zum Aufräumen: rm ${lock_file} ${pid_file}"
            fi
        else
            echo "  ⚠️  ${name}: Lock-File ohne PID"
        fi
    else
        echo "  ⭕ ${name}: Nicht aktiv"
    fi
}

check_lock "chain"
check_lock "person"

echo ""
echo "==============================================================================="

# ============================================================================
# Laufende Python-Prozesse
# ============================================================================
echo ""
echo "🐍 Laufende Python-Prozesse (person.py):"
echo ""

PYTHON_PROCS=$(ps aux | grep "[p]erson.py" | grep -v grep)

if [ -z "$PYTHON_PROCS" ]; then
    echo "  ⭕ Keine person.py Prozesse laufen"
else
    echo "$PYTHON_PROCS" | awk '{printf "  PID: %-7s CPU: %-5s MEM: %-5s Zeit: %-10s\n", $2, $3"%", $4"%", $10}'

    # Zähle Prozesse
    COUNT=$(echo "$PYTHON_PROCS" | wc -l)
    if [ "$COUNT" -gt 1 ]; then
        echo ""
        echo "  ⚠️  WARNUNG: $COUNT parallele Prozesse gefunden!"
        echo "  Das führt zu CUDA Out of Memory!"
    fi
fi

echo ""
echo "==============================================================================="

# ============================================================================
# GPU Status
# ============================================================================
echo ""
echo "🎮 GPU Status:"
echo ""

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    while IFS=, read -r idx name mem_used mem_total gpu_util; do
        mem_percent=$((mem_used * 100 / mem_total))
        echo "  GPU ${idx}: ${name}"
        echo "    Memory: ${mem_used} MB / ${mem_total} MB (${mem_percent}%)"
        echo "    GPU:    ${gpu_util}%"
    done

    echo ""
    echo "  Prozesse auf GPU:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | \
    while IFS=, read -r pid mem; do
        cmd=$(ps -p ${pid} -o comm= 2>/dev/null || echo "unknown")
        echo "    PID ${pid}: ${mem} MB (${cmd})"
    done
else
    echo "  ⚠️  nvidia-smi nicht verfügbar"
fi

echo ""
echo "==============================================================================="

# ============================================================================
# Letzte Logs
# ============================================================================
echo ""
echo "📊 Letzte Log-Einträge (chain.log):"
echo ""

if [ -f "${SCRIPT_DIR}/logs/chain.log" ]; then
    tail -15 "${SCRIPT_DIR}/logs/chain.log" | sed 's/^/  /'
else
    echo "  ⭕ Keine chain.log gefunden"
fi

echo ""
echo "==============================================================================="

# ============================================================================
# Aktionen
# ============================================================================
echo ""
echo "🔧 Verfügbare Aktionen:"
echo ""
echo "  Alle Python-Prozesse stoppen:"
echo "    killall python3"
echo ""
echo "  Stale Locks entfernen:"
echo "    rm ${SCRIPT_DIR}/.*.lock ${SCRIPT_DIR}/.*.pid"
echo ""
echo "  Chain starten:"
echo "    ${SCRIPT_DIR}/run_chain.sh"
echo ""
echo "  Logs live anschauen:"
echo "    tail -f ${SCRIPT_DIR}/logs/chain.log"
echo ""
echo "==============================================================================="
