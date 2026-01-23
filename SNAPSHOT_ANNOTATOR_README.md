# Snapshot-Annotator

Erstellt automatisch jede Minute ein annotiertes Bild mit YOLO-Objekterkennung von deiner Reolink-Kamera.

## Features

- 📸 **Automatische Snapshots** alle 60 Sekunden (konfigurierbar)
- 🤖 **YOLO-Analyse** mit GPU-Beschleunigung (Tesla P4)
- 🎨 **Annotierte Bilder** mit Bounding Boxes:
  - 🚗 Gelbe Boxen: Fahrzeuge
  - 👤 Grüne Boxen: Personen
  - 😊 Gelbe Boxen: Bekannte Gesichter
  - 😐 Rote Boxen: Unbekannte Gesichter
- 🧹 **Auto-Cleanup**: Alte Bilder werden automatisch gelöscht (Standard: 24h)
- 🔄 **Systemd-Service**: Läuft als Daemon im Hintergrund

## Installation

### 1. Konfiguration anpassen

Öffne `snapshot_annotator.py` und setze:

```python
CAMERA_IP = "192.168.178.123"     # Deine Kamera IP
CAMERA_USER = "admin"              # Kamera Benutzername
CAMERA_PASS = "dein_passwort"      # Kamera Passwort
```

### 2. Setup ausführen

```bash
# Basis-Setup
./setup_snapshot_annotator.sh

# Mit systemd-Service (als root)
sudo ./setup_snapshot_annotator.sh
```

### 3. Test

```bash
# Manueller Test (stoppt nach Ctrl+C)
./snapshot_annotator.py
```

### 4. Service starten

```bash
# Service starten
sudo systemctl start snapshot-annotator

# Service-Status prüfen
sudo systemctl status snapshot-annotator

# Logs ansehen
sudo journalctl -u snapshot-annotator -f
```

## Verzeichnisse

```
/var/www/web1/snapshots/           - Original-Snapshots
/var/www/web1/snapshots_annotated/ - Annotierte Bilder mit Boxen
./logs/snapshot_annotator.log      - Log-Datei
```

## Konfiguration

### Intervall ändern

```bash
# Alle 30 Sekunden
./snapshot_annotator.py --interval 30

# Alle 5 Minuten (300 Sekunden)
./snapshot_annotator.py --interval 300
```

### Aufbewahrungszeit ändern

```bash
# Bilder 48 Stunden behalten
./snapshot_annotator.py --keep-hours 48

# Nur 6 Stunden behalten
./snapshot_annotator.py --keep-hours 6
```

## Service-Befehle

```bash
# Starten
sudo systemctl start snapshot-annotator

# Stoppen
sudo systemctl stop snapshot-annotator

# Neu starten
sudo systemctl restart snapshot-annotator

# Status prüfen
sudo systemctl status snapshot-annotator

# Auto-Start aktivieren
sudo systemctl enable snapshot-annotator

# Auto-Start deaktivieren
sudo systemctl disable snapshot-annotator

# Logs (live)
sudo journalctl -u snapshot-annotator -f

# Logs (letzte 100 Zeilen)
sudo journalctl -u snapshot-annotator -n 100
```

## Beispiel-Output

```
2026-01-23 21:30:00 - INFO - [1] 2026-01-23 21:30:00
2026-01-23 21:30:02 - INFO - ✓ Snapshot gespeichert: Camera1_20260123_213000.jpg
2026-01-23 21:30:02 - INFO - 🔍 Analysiere Camera1_20260123_213000.jpg...
2026-01-23 21:30:03 - INFO -   Erkannt: 2 Fahrzeuge, 1 Personen, 1 Gesichter (1 bekannt)
2026-01-23 21:30:03 - INFO - ✓ Annotiert: annotated_Camera1_Camera1_20260123_213000.jpg
2026-01-23 21:30:03 - INFO - ⏸ Warte 60s bis nächster Snapshot...
```

## Annotierte Bilder ansehen

```bash
# Liste aller annotierten Bilder
ls -lht /var/www/web1/snapshots_annotated/ | head -20

# Neuestes Bild ansehen (mit xdg-open)
xdg-open /var/www/web1/snapshots_annotated/$(ls -t /var/www/web1/snapshots_annotated/ | head -1)
```

## Web-Zugriff

Wenn `/var/www/web1` über einen Webserver erreichbar ist:

```
http://deine-server-ip/snapshots_annotated/
```

## Troubleshooting

### Kamera nicht erreichbar

```bash
# Teste Kamera-Verbindung
curl -u admin:passwort "http://192.168.178.123/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=snapshot" -o test.jpg
```

### GPU nicht verfügbar

Das Script läuft auch auf CPU, ist aber langsamer. Prüfe GPU:

```bash
nvidia-smi
```

### Logs prüfen

```bash
# Script-Logs
tail -f ./logs/snapshot_annotator.log

# Systemd-Logs
sudo journalctl -u snapshot-annotator -f
```

### Service startet nicht

```bash
# Status prüfen
sudo systemctl status snapshot-annotator

# Service neu laden
sudo systemctl daemon-reload
sudo systemctl restart snapshot-annotator
```

## Performance

- **CPU**: ~2-5 Sekunden pro Bild
- **GPU (Tesla P4)**: ~0.5-1 Sekunde pro Bild
- **Speicher**: ~100-200 MB pro Bild
- **Platz (24h)**: ~15-20 GB (bei 4K-Bildern, 1440 Bilder)

## Deinstallation

```bash
# Service stoppen und deaktivieren
sudo systemctl stop snapshot-annotator
sudo systemctl disable snapshot-annotator
sudo rm /etc/systemd/system/snapshot-annotator.service
sudo systemctl daemon-reload

# Bilder löschen (optional)
rm -rf /var/www/web1/snapshots*
```

## Support

Bei Problemen:
1. Logs prüfen: `sudo journalctl -u snapshot-annotator -f`
2. Manuell testen: `./snapshot_annotator.py --interval 60`
3. GPU prüfen: `nvidia-smi`
4. Kamera-Verbindung testen: `curl ...`
