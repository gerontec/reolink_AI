# Annotierte Bilder mit watchdog2.py

Die **einfachste Lösung**: watchdog2.py hat bereits einen ImageAnnotator eingebaut!

## Setup

### 1. Watchdog2.py mit Annotation starten

```bash
# Mit Annotation (erstellt annotierte Bilder für ALLE analysierten Bilder)
./watchdog2.py --save-annotated

# Oder nur die letzten 10 Dateien
./watchdog2.py --save-annotated --limit 10
```

### 2. Wo sind die Bilder?

**Annotierte Bilder:**
```bash
ls -lht /var/www/web1/annotated/ | head -20
```

**Neuestes annotiertes Bild (per Kamera):**
```bash
# Watchdog2.py erstellt automatisch "latest_CameraX.jpg"
ls -l /var/www/web1/annotated/latest_*.jpg
```

## Automatisch im Hintergrund

### Option A: Watchdog2.sh anpassen

Bearbeite `/home/gh/python/reolink_AI/watchdog2.sh`:

```bash
#!/bin/bash
cd /home/gh/python/reolink_AI

# Füge --save-annotated hinzu
./watchdog2.py --save-annotated --limit 100
```

### Option B: Systemd Service

Falls watchdog2.py als Service läuft, Service-File anpassen:

```bash
# Service-File öffnen
sudo nano /etc/systemd/system/watchdog2.service

# In ExecStart Zeile --save-annotated hinzufügen:
ExecStart=/home/gh/python/reolink_AI/watchdog2.py --save-annotated

# Service neu laden
sudo systemctl daemon-reload
sudo systemctl restart watchdog2
```

### Option C: Cron-Job (alle 5 Minuten)

```bash
# Crontab öffnen
crontab -e

# Folgende Zeile hinzufügen (alle 5 Minuten)
*/5 * * * * cd /home/gh/python/reolink_AI && ./watchdog2.py --save-annotated --limit 5 >> ./logs/watchdog_cron.log 2>&1
```

## Neuestes Bild immer verfügbar

Watchdog2.py erstellt automatisch:

```
/var/www/web1/annotated/latest_Camera1.jpg
/var/www/web1/annotated/latest_Camera2.jpg
```

Diese Dateien werden **immer überschrieben** mit dem neuesten annotierten Bild.

## Web-Zugriff

Wenn `/var/www/web1` über einen Webserver erreichbar ist:

```
http://deine-server-ip/annotated/latest_Camera1.jpg
http://deine-server-ip/annotated/
```

## Einfacher HTML-Viewer

Erstelle `annotated_viewer.html`:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Live Annotated View</title>
    <meta http-equiv="refresh" content="60">
</head>
<body style="margin:0; padding:20px; background:#000; text-align:center;">
    <h1 style="color:#fff;">Live Annotated Camera Feed</h1>
    <img src="/annotated/latest_Camera1.jpg"
         style="max-width:100%; border:2px solid #0f0;"
         onerror="this.src='/annotated/latest_Camera1.jpg?'+new Date().getTime()">
    <p style="color:#888; margin-top:20px;">
        Auto-refresh alle 60 Sekunden |
        Letzte Aktualisierung: <span id="time"></span>
    </p>
    <script>
        document.getElementById('time').textContent = new Date().toLocaleString('de-DE');
    </script>
</body>
</html>
```

## Vorteile dieser Methode

✅ Nutzt vorhandenen watchdog2.py Code (keine zusätzlichen Scripts)
✅ Keine extra Dependencies nötig (alles schon installiert)
✅ Läuft auf dem gleichen Python venv wie watchdog2.py
✅ Funktioniert mit Videos UND Bildern
✅ `latest_CameraX.jpg` wird automatisch erstellt
✅ Alte annotierte Bilder bleiben erhalten (falls gewünscht)

## Troubleshooting

### Keine annotierten Bilder?

```bash
# Prüfe ob --save-annotated gesetzt ist
ps aux | grep watchdog2.py

# Teste manuell
./watchdog2.py --save-annotated --limit 1
```

### Verzeichnis existiert nicht?

```bash
sudo mkdir -p /var/www/web1/annotated
sudo chown -R gh:gh /var/www/web1/annotated
```

### Latest.jpg wird nicht erstellt?

Prüfe watchdog2.py Logs:
```bash
tail -f ./logs/watchdog.log | grep "Latest-Kopie"
```

## Cleanup alte Bilder (optional)

```bash
# Lösche annotierte Bilder älter als 7 Tage
find /var/www/web1/annotated/ -name "*.jpg" ! -name "latest_*" -mtime +7 -delete
```

## Fazit

**Einfachste Lösung:**
```bash
./watchdog2.py --save-annotated
```

Fertig! 🎉

Das erstellt automatisch:
- Annotierte Bilder für alle analysierten Videos/Bilder
- `latest_CameraX.jpg` das immer das neueste Bild zeigt
- Bounding Boxes für Fahrzeuge, Personen, Gesichter
