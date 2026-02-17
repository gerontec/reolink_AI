#!/bin/bash
#
# Installation Script für Reolink AI Service
# Vereinfachtes Service-Konzept: 1 Timer + 1 Service
#

set -e

echo "=========================================="
echo "  Reolink AI Service Installation"
echo "=========================================="
echo ""

# Farben
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}✗ Bitte als root ausführen: sudo $0${NC}"
    exit 1
fi

echo "1. Alte Services stoppen und deaktivieren..."
for service in video-analyzer video-recorder watchdog-cam; do
    if systemctl is-active --quiet ${service}.service 2>/dev/null; then
        echo -e "  ${YELLOW}Stoppe ${service}...${NC}"
        systemctl stop ${service}.service || true
    fi
    if systemctl is-enabled --quiet ${service}.service 2>/dev/null; then
        echo -e "  ${YELLOW}Deaktiviere ${service}...${NC}"
        systemctl disable ${service}.service || true
    fi
done

echo ""
echo "2. Service-Dateien installieren..."
cp reolink-ai.service /etc/systemd/system/
cp reolink-ai.timer /etc/systemd/system/
chmod 644 /etc/systemd/system/reolink-ai.service
chmod 644 /etc/systemd/system/reolink-ai.timer
echo -e "  ${GREEN}✓ Service-Dateien kopiert${NC}"

echo ""
echo "3. Systemd neu laden..."
systemctl daemon-reload
echo -e "  ${GREEN}✓ Daemon reloaded${NC}"

echo ""
echo "4. Service aktivieren..."
systemctl enable reolink-ai.timer
echo -e "  ${GREEN}✓ Timer aktiviert${NC}"

echo ""
echo "5. Service starten..."
systemctl start reolink-ai.timer
echo -e "  ${GREEN}✓ Timer gestartet${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ Installation abgeschlossen!${NC}"
echo "=========================================="
echo ""
echo "Status prüfen:"
echo "  systemctl status reolink-ai.timer"
echo "  systemctl list-timers reolink-ai.timer"
echo ""
echo "Logs ansehen:"
echo "  journalctl -u reolink-ai.service -f"
echo ""
echo "Service manuell ausführen:"
echo "  systemctl start reolink-ai.service"
echo ""
echo "Konfiguration:"
echo "  - Läuft alle 2 Minuten"
echo "  - Verarbeitet max. 50 Dateien pro Durchlauf"
echo "  - GPU-beschleunigt (CUDA)"
echo "  - Auto-Restart bei Fehlern"
echo ""
