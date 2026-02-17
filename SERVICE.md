# Reolink AI - Vereinfachtes Cron-Setup

## Überblick

**Maximal einfach: Nur Crontab!**

```
Crontab (alle 2 Min) ──> person.py --limit 50 ──> GPU ──> DB
```

## Installation

```bash
cd /home/user/reolink_AI
sudo ./setup-cron.sh
```

Das Script:
1. ✅ Stoppt ALLE alten Services (video-analyzer, video-recorder, watchdog-cam, reolink-ai)
2. ✅ Deaktiviert alle Services
3. ✅ Erstellt einen einzigen Crontab-Eintrag für User 'gh'
4. ✅ Richtet Logging ein (/var/log/reolink-ai.log)
5. ✅ Konfiguriert Logrotate (7 Tage)

## Crontab-Eintrag

```cron
# Alle 2 Minuten
*/2 * * * * cd /home/user/reolink_AI && /usr/bin/python3 person.py --limit 50 >> /var/log/reolink-ai.log 2>&1
```

## Warum nur Cron?

✅ **GPU macht es möglich:**
- 50 Dateien in ~2 Minuten
- Keine Daemons nötig
- Einfach, robust, bewährt

✅ **Vorteile:**
- Keine systemd-Komplexität
- Keine Service-Dependencies
- Automatischer Neustart bei Fehler (Cron)
- Einfaches Debugging (Logfile)

## Verwaltung

### Crontab anzeigen
```bash
crontab -u gh -l
```

### Crontab bearbeiten
```bash
crontab -u gh -e
```

### Intervall ändern
```cron
*/1 * * * *   # Jede Minute
*/2 * * * *   # Alle 2 Minuten (Standard)
*/5 * * * *   # Alle 5 Minuten
*/10 * * * *  # Alle 10 Minuten
```

### Logs ansehen
```bash
# Live-Logs
tail -f /var/log/reolink-ai.log

# Letzte 100 Zeilen
tail -n 100 /var/log/reolink-ai.log

# Suche nach Fehlern
grep -i error /var/log/reolink-ai.log
```

### Batch-Size anpassen
```bash
crontab -u gh -e
```

Ändere `--limit 50` zu gewünschter Größe:
- `--limit 25` = Weniger Last
- `--limit 100` = Mehr Durchsatz

### Cron-Job manuell testen
```bash
su - gh
cd /home/user/reolink_AI
python3 person.py --limit 50
```

## Performance

**GPU-Beschleunigung (Tesla P4):**
- 50 Dateien in ~2 Minuten = 2.4s/Datei
- Inkl. Face Recognition (InsightFace GPU)
- Inkl. Object Detection (YOLO GPU)
- Inkl. Best-Frame-Extraktion (MP4 → JPG)

**Timing:**
```
Alle 2 Minuten:   Optimal für normale Last
Alle 1 Minute:    Für hohen Durchsatz
Alle 5 Minuten:   Für wenig Aktivität
```

## Migration von Services

Das Setup-Script macht alles automatisch:

```bash
sudo ./setup-cron.sh
```

**Manuell:**
```bash
# Services stoppen
sudo systemctl stop video-analyzer.service
sudo systemctl stop video-recorder.service
sudo systemctl stop watchdog-cam.service
sudo systemctl stop reolink-ai.timer

# Services deaktivieren
sudo systemctl disable video-analyzer.service
sudo systemctl disable video-recorder.service
sudo systemctl disable watchdog-cam.service
sudo systemctl disable reolink-ai.timer

# Crontab einrichten
sudo crontab -u gh -e
# Eintrag hinzufügen (siehe oben)
```

## Alte Services

| Service | Status | Grund |
|---------|--------|-------|
| video-analyzer.service | ✗ Deaktiviert | Ersetzt durch Cron |
| video-recorder.service | ✗ Deaktiviert | Ersetzt durch Cron |
| watchdog-cam.service | ✗ Deaktiviert | Ersetzt durch Cron |
| reolink-ai.timer | ✗ Deaktiviert | Ersetzt durch Cron |
| watchdog-mux.service | ✓ Bleibt | Proxmox-spezifisch |

## Troubleshooting

### Cron läuft nicht
```bash
# Cron-Daemon prüfen
systemctl status cron

# Cron-Daemon starten
sudo systemctl start cron

# Crontab prüfen
crontab -u gh -l
```

### Keine Logs
```bash
# Log-Datei prüfen
ls -la /var/log/reolink-ai.log

# Neu erstellen
sudo touch /var/log/reolink-ai.log
sudo chown gh:gh /var/log/reolink-ai.log
```

### GPU nicht gefunden
```bash
# Als User 'gh' testen
su - gh
python3 -c "import torch; print(torch.cuda.is_available())"

# CUDA-Pfad in Crontab setzen (falls nötig)
crontab -u gh -e
# Am Anfang hinzufügen:
PATH=/usr/local/cuda/bin:/usr/bin:/bin
LD_LIBRARY_PATH=/usr/local/cuda/lib64
```

### MySQL Connection Error
```bash
# MySQL-Socket prüfen (Cron hat andere Umgebung!)
ls -la /var/run/mysqld/mysqld.sock

# In person.py DB_CONFIG anpassen (falls nötig):
# 'unix_socket': '/var/run/mysqld/mysqld.sock'
```

## Monitoring

### Letzte Ausführung
```bash
tail -n 50 /var/log/reolink-ai.log | grep "Verarbeitung abgeschlossen"
```

### Statistik
```bash
# Wie viele Dateien heute?
grep "$(date +%Y-%m-%d)" /var/log/reolink-ai.log | grep "verarbeitet" | wc -l

# Fehler heute?
grep "$(date +%Y-%m-%d)" /var/log/reolink-ai.log | grep -i error
```

### Datenbank-Report
```bash
python3 cam2_report.py
```

### GPU-Auslastung
```bash
watch -n 1 nvidia-smi
```

## Backup

```bash
# Crontab sichern
crontab -u gh -l > /home/user/reolink_AI/crontab-backup.txt

# Crontab wiederherstellen
crontab -u gh /home/user/reolink_AI/crontab-backup.txt
```

## Deinstallation

```bash
# Crontab-Eintrag entfernen
crontab -u gh -e
# Zeile mit person.py löschen

# Oder komplett löschen:
crontab -u gh -r
```
