#!/bin/bash
# Setup-Script für Snapshot-Annotator

echo "================================================"
echo "Snapshot-Annotator Setup"
echo "================================================"

# Prüfe ob wir root sind (für systemd)
if [ "$EUID" -ne 0 ]; then
    echo "⚠ Warnung: Nicht als root ausgeführt"
    echo "   Für systemd-Service Installation: sudo ./setup_snapshot_annotator.sh"
fi

# 1. Verzeichnisse erstellen
echo ""
echo "1. Erstelle Verzeichnisse..."
mkdir -p /var/www/web1/snapshots
mkdir -p /var/www/web1/snapshots_annotated
mkdir -p ./logs
chown -R gh:gh /var/www/web1/snapshots /var/www/web1/snapshots_annotated
echo "✓ Verzeichnisse erstellt"

# 2. Script ausführbar machen
echo ""
echo "2. Mache Scripts ausführbar..."
chmod +x snapshot_annotator.py
echo "✓ Berechtigungen gesetzt"

# 3. Konfiguration prüfen
echo ""
echo "3. Prüfe Konfiguration..."
if grep -q "192.168.178.xxx" snapshot_annotator.py; then
    echo "⚠ WARNUNG: Kamera-IP noch nicht konfiguriert!"
    echo ""
    echo "Bitte bearbeite snapshot_annotator.py und setze:"
    echo "  - CAMERA_IP (Zeile 33)"
    echo "  - CAMERA_USER (Zeile 34)"
    echo "  - CAMERA_PASS (Zeile 35)"
    echo ""
    read -p "Möchtest du die Datei jetzt bearbeiten? (j/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Jj]$ ]]; then
        nano snapshot_annotator.py
    fi
fi

# 4. Test-Lauf anbieten
echo ""
echo "4. Test-Lauf (optional)..."
read -p "Möchtest du einen Test-Lauf machen? (j/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Jj]$ ]]; then
    echo ""
    echo "Starte Test (1 Snapshot)..."
    echo "Drücke Ctrl+C nach dem ersten Snapshot zum Abbrechen"
    echo ""
    ./snapshot_annotator.py --interval 60
fi

# 5. Systemd-Service installieren (nur als root)
if [ "$EUID" -eq 0 ]; then
    echo ""
    echo "5. Systemd-Service installieren..."
    read -p "Soll der Service automatisch beim Systemstart starten? (j/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Jj]$ ]]; then
        cp snapshot-annotator.service /etc/systemd/system/
        systemctl daemon-reload
        systemctl enable snapshot-annotator.service
        echo "✓ Service installiert und aktiviert"
        echo ""
        echo "Service-Befehle:"
        echo "  Start:   sudo systemctl start snapshot-annotator"
        echo "  Stop:    sudo systemctl stop snapshot-annotator"
        echo "  Status:  sudo systemctl status snapshot-annotator"
        echo "  Logs:    sudo journalctl -u snapshot-annotator -f"
    fi
else
    echo ""
    echo "5. Systemd-Service (übersprungen - nicht als root)"
    echo "   Führe 'sudo ./setup_snapshot_annotator.sh' aus für Service-Installation"
fi

echo ""
echo "================================================"
echo "✓ Setup abgeschlossen!"
echo "================================================"
echo ""
echo "Nächste Schritte:"
echo "  1. Konfiguriere Kamera-Zugangsdaten in snapshot_annotator.py"
echo "  2. Test: ./snapshot_annotator.py"
echo "  3. Service starten: sudo systemctl start snapshot-annotator"
echo "  4. Bilder ansehen: ls -lh /var/www/web1/snapshots_annotated/"
echo ""
