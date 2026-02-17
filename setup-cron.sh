#!/bin/bash
#
# Reolink AI - Crontab Setup
# Deaktiviert alle Services, nutzt nur Cron
#

set -e

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "  Reolink AI - Crontab Setup"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}✗ Bitte als root ausführen: sudo $0${NC}"
    exit 1
fi

echo "1. Alle alten Services stoppen und deaktivieren..."
for service in video-analyzer video-recorder watchdog-cam reolink-ai.timer reolink-ai; do
    if systemctl is-active --quiet ${service}.service 2>/dev/null || systemctl is-active --quiet ${service} 2>/dev/null; then
        echo -e "  ${YELLOW}Stoppe ${service}...${NC}"
        systemctl stop ${service}.service 2>/dev/null || systemctl stop ${service} 2>/dev/null || true
    fi
    if systemctl is-enabled --quiet ${service}.service 2>/dev/null || systemctl is-enabled --quiet ${service} 2>/dev/null; then
        echo -e "  ${YELLOW}Deaktiviere ${service}...${NC}"
        systemctl disable ${service}.service 2>/dev/null || systemctl disable ${service} 2>/dev/null || true
    fi
done
echo -e "  ${GREEN}✓ Alle Services gestoppt${NC}"

echo ""
echo "2. Crontab für User 'gh' einrichten..."

# Erkenne venv
VENV_PYTHON=""
if [ -f "/home/gh/python/reolink_AI/venv_py311/bin/python3" ]; then
    VENV_PYTHON="/home/gh/python/reolink_AI/venv_py311/bin/python3"
    echo -e "  ${GREEN}✓ venv gefunden: venv_py311${NC}"
elif [ -f "/home/user/reolink_AI/venv/bin/python3" ]; then
    VENV_PYTHON="/home/user/reolink_AI/venv/bin/python3"
    echo -e "  ${GREEN}✓ venv gefunden: venv${NC}"
else
    VENV_PYTHON="/usr/bin/python3"
    echo -e "  ${YELLOW}⚠ Kein venv gefunden, nutze System-Python${NC}"
fi

# Crontab-Eintrag mit venv-Python
CRON_ENTRY="*/2 * * * * cd /home/gh/python/reolink_AI && ${VENV_PYTHON} person.py --limit 50 >> /var/log/reolink-ai.log 2>&1"

# Prüfen ob Eintrag bereits existiert
if crontab -u gh -l 2>/dev/null | grep -q "person.py"; then
    echo -e "  ${YELLOW}Crontab-Eintrag existiert bereits, wird aktualisiert...${NC}"
    # Entferne alte Einträge
    crontab -u gh -l 2>/dev/null | grep -v "person.py" | crontab -u gh -
fi

# Neuen Eintrag hinzufügen
(crontab -u gh -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -u gh -

echo -e "  ${GREEN}✓ Crontab eingerichtet${NC}"

echo ""
echo "3. Log-Datei vorbereiten..."
touch /var/log/reolink-ai.log
chown gh:gh /var/log/reolink-ai.log
chmod 644 /var/log/reolink-ai.log
echo -e "  ${GREEN}✓ Log-Datei erstellt${NC}"

echo ""
echo "4. Logrotate konfigurieren..."
cat > /etc/logrotate.d/reolink-ai <<'LOGROTATE'
/var/log/reolink-ai.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 gh gh
}
LOGROTATE
echo -e "  ${GREEN}✓ Logrotate konfiguriert${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Setup abgeschlossen!${NC}"
echo "=========================================="
echo ""
echo "Crontab anzeigen:"
echo "  crontab -u gh -l"
echo ""
echo "Logs ansehen:"
echo "  tail -f /var/log/reolink-ai.log"
echo ""
echo "Konfiguration:"
echo "  - Läuft alle 2 Minuten"
echo "  - Verarbeitet max. 50 Dateien pro Durchlauf"
echo "  - GPU-beschleunigt"
echo "  - Logs in /var/log/reolink-ai.log"
echo ""
echo "Intervall ändern:"
echo "  crontab -u gh -e"
echo "  */2 = alle 2 Minuten"
echo "  */5 = alle 5 Minuten"
echo "  * = jede Minute"
echo ""
