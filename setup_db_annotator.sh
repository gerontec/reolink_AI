#!/bin/bash
# Setup-Script für DB-Annotator

echo "================================================"
echo "DB-Annotator Setup"
echo "================================================"

# 1. Python-Dependencies prüfen
echo ""
echo "1. Prüfe Python-Dependencies..."

python3 -c "import pymysql" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ pymysql nicht gefunden - installiere..."
    pip3 install pymysql
fi

python3 -c "import cv2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ opencv-python nicht gefunden - installiere..."
    pip3 install opencv-python
fi

python3 -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ numpy nicht gefunden - installiere..."
    pip3 install numpy
fi

echo "✓ Dependencies OK"

# 2. Verzeichnisse erstellen
echo ""
echo "2. Erstelle Verzeichnisse..."
mkdir -p /var/www/web1/annotated_from_db
mkdir -p ./logs
chown -R gh:gh /var/www/web1/annotated_from_db 2>/dev/null || true
echo "✓ Verzeichnisse erstellt"

# 3. Script ausführbar machen
echo ""
echo "3. Setze Berechtigungen..."
chmod +x db_annotator.py
echo "✓ Berechtigungen gesetzt"

# 4. Test-Lauf
echo ""
echo "4. Test-Lauf (optional)..."
read -p "Möchtest du einen Test-Lauf machen? (j/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Jj]$ ]]; then
    echo ""
    echo "Starte Test (Ctrl+C zum Abbrechen)..."
    echo ""
    ./db_annotator.py --interval 60
fi

# 5. Systemd-Service (nur als root)
if [ "$EUID" -eq 0 ]; then
    echo ""
    echo "5. Systemd-Service installieren..."
    read -p "Soll der Service automatisch starten? (j/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Jj]$ ]]; then
        cp db-annotator.service /etc/systemd/system/
        systemctl daemon-reload
        systemctl enable db-annotator
        echo "✓ Service installiert und aktiviert"
        echo ""
        echo "Service-Befehle:"
        echo "  Start:   sudo systemctl start db-annotator"
        echo "  Stop:    sudo systemctl stop db-annotator"
        echo "  Status:  sudo systemctl status db-annotator"
        echo "  Logs:    sudo journalctl -u db-annotator -f"
    fi
else
    echo ""
    echo "5. Systemd-Service (übersprungen - nicht als root)"
    echo "   Führe 'sudo ./setup_db_annotator.sh' aus für Service-Installation"
fi

echo ""
echo "================================================"
echo "✓ Setup abgeschlossen!"
echo "================================================"
echo ""
echo "Nächste Schritte:"
echo "  1. Test: ./db_annotator.py"
echo "  2. Service starten: sudo systemctl start db-annotator"
echo "  3. Bilder ansehen: ls -lh /var/www/web1/annotated_from_db/"
echo ""
