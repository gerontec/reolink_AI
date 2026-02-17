#!/bin/bash
################################################################################
# Log Cleanup Script
# Löscht alte Log-Dateien (behaltet nur .log und .log.old)
################################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

echo "==============================================================================="
echo "  Log Cleanup"
echo "==============================================================================="
echo ""
echo "Log-Verzeichnis: ${LOG_DIR}"
echo ""

# Prüfe ob Verzeichnis existiert
if [ ! -d "${LOG_DIR}" ]; then
    echo "❌ Log-Verzeichnis existiert nicht: ${LOG_DIR}"
    exit 1
fi

cd "${LOG_DIR}"

# Zeige aktuelle Log-Dateien
echo "Aktuelle Logs (werden BEHALTEN):"
echo "  ✅ person.log"
echo "  ✅ person.log.old"
echo "  ✅ cluster.log"
echo "  ✅ cluster.log.old"
echo "  ✅ chain.log"
echo "  ✅ chain.log.old"
echo ""

# Finde alte Logs (mit Timestamp im Namen)
OLD_LOGS=$(find . -maxdepth 1 -name "*.log" -type f \
    ! -name "person.log" \
    ! -name "person.log.old" \
    ! -name "cluster.log" \
    ! -name "cluster.log.old" \
    ! -name "chain.log" \
    ! -name "chain.log.old" \
    2>/dev/null)

if [ -z "$OLD_LOGS" ]; then
    echo "✅ Keine alten Logs gefunden - alles sauber!"
    exit 0
fi

# Zeige zu löschende Dateien
echo "Folgende alte Logs werden GELÖSCHT:"
echo ""
echo "$OLD_LOGS" | while read -r file; do
    SIZE=$(du -h "$file" | cut -f1)
    echo "  🗑️  $file ($SIZE)"
done

# Zähle Dateien
COUNT=$(echo "$OLD_LOGS" | wc -l)
TOTAL_SIZE=$(echo "$OLD_LOGS" | xargs du -ch 2>/dev/null | tail -1 | cut -f1)

echo ""
echo "Gesamt: $COUNT Dateien, $TOTAL_SIZE"
echo ""

# Sicherheitsabfrage
read -p "Wirklich löschen? [y/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Lösche alte Logs..."
    echo "$OLD_LOGS" | xargs rm -f
    echo ""
    echo "✅ Cleanup abgeschlossen!"
else
    echo ""
    echo "❌ Abgebrochen - keine Dateien gelöscht"
    exit 1
fi

echo ""
echo "Verbleibende Logs:"
ls -lh *.log *.log.old 2>/dev/null || echo "  (keine)"
echo ""
echo "==============================================================================="
