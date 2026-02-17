# Reolink AI Service - Vereinfachtes Konzept

## Überblick

**Alles in einem Service!** Statt 3 separater Services jetzt nur noch:

```
reolink-ai.timer  ──> Läuft alle 2 Minuten
       │
       └──> reolink-ai.service ──> person.py --limit 50
```

## Warum nur ein Service?

✅ **GPU-Beschleunigung macht es möglich:**
- Face Detection: ~0.2s (vorher: 1.5s auf CPU)
- YOLO Detection: ~0.1s (vorher: 0.3s)
- **7x schneller** → Echtzeit-Verarbeitung nicht mehr nötig!

## Installation

```bash
cd /home/user/reolink_AI
sudo ./install-service.sh
```

Das Script:
1. Stoppt alte Services (video-analyzer, video-recorder, watchdog-cam)
2. Installiert neuen Timer + Service
3. Aktiviert und startet den Service

## Konfiguration

### Timer-Intervall anpassen

```bash
sudo systemctl edit reolink-ai.timer
```

```ini
[Timer]
OnUnitActiveSec=5min  # Alle 5 Minuten statt 2
```

### Batch-Size anpassen

```bash
sudo systemctl edit reolink-ai.service
```

```ini
[Service]
ExecStart=/usr/bin/python3 /home/user/reolink_AI/person.py --limit 100
```

## Verwaltung

### Status prüfen
```bash
systemctl status reolink-ai.timer
systemctl list-timers reolink-ai.timer
```

### Logs ansehen
```bash
# Live-Logs
journalctl -u reolink-ai.service -f

# Letzte 100 Zeilen
journalctl -u reolink-ai.service -n 100
```

### Manuell ausführen
```bash
sudo systemctl start reolink-ai.service
```

### Service stoppen
```bash
sudo systemctl stop reolink-ai.timer
```

### Service deaktivieren
```bash
sudo systemctl disable reolink-ai.timer
```

## Performance

**GPU-Beschleunigung (Tesla P4):**
- 50 Dateien in ~2 Minuten
- ~2.4s pro Datei (inkl. DB-Zugriff)
- Vollautomatisch, keine Verzögerung

**Timer-Interval:**
- Standard: 2 Minuten
- Bei hohem Durchsatz: 1 Minute
- Bei wenig Aktivität: 5 Minuten

## Alte Services

Die alten Services sind **nicht mehr nötig**:

| Alt | Neu | Grund |
|-----|-----|-------|
| video-analyzer.service | ✗ | Integriert in reolink-ai |
| video-recorder.service | ✗ | Integriert in reolink-ai |
| watchdog-cam.service | ✗ | Integriert in reolink-ai |
| watchdog-mux.service | ✓ | Proxmox-spezifisch, bleibt |

## Troubleshooting

### Service läuft nicht
```bash
journalctl -u reolink-ai.service -n 50
```

### GPU nicht erkannt
```bash
nvidia-smi
# Falls Fehler: CUDA-Treiber installieren
```

### MySQL Connection Error
```bash
systemctl status mysql
# Service starten falls nötig:
systemctl start mysql
```

### Zu viele Dateien
```bash
# Batch-Size erhöhen
sudo systemctl edit reolink-ai.service
# ExecStart mit --limit 100 oder höher
```

## Migration von alten Services

```bash
# Alte Services stoppen
sudo systemctl stop video-analyzer.service
sudo systemctl stop video-recorder.service
sudo systemctl stop watchdog-cam.service

# Deaktivieren (nicht mehr auto-starten)
sudo systemctl disable video-analyzer.service
sudo systemctl disable video-recorder.service
sudo systemctl disable watchdog-cam.service

# Neuen Service installieren
cd /home/user/reolink_AI
sudo ./install-service.sh
```

## Monitoring

```bash
# Wie viele Dateien wurden verarbeitet?
journalctl -u reolink-ai.service | grep "verarbeitet"

# GPU-Auslastung überwachen
watch -n 1 nvidia-smi

# Datenbank-Report
python3 cam2_report.py
```
