# Services & Cron

## Aktive Prozesse

### cam2-stream.service (systemd, dauerhaft)

```bash
sudo systemctl start   cam2-stream.service
sudo systemctl stop    cam2-stream.service
sudo systemctl restart cam2-stream.service
sudo systemctl status  cam2-stream.service
journalctl -u cam2-stream -f
```

Funktion: Liest Cam2 Sub-Stream (640×360), erkennt Personen via YOLO,
triggert bei Erkennung eine 25s Aufnahme vom Main-Stream (1920×1080).

Konfiguration in `cam2_stream.py`:
- `CLIP_DURATION = 25` — Sekunden je Aufnahme
- `CLIP_COOLDOWN = 90` — Mindestpause zwischen Aufnahmen
- `CONF_THRESHOLD = 0.45` — YOLO-Schwellwert
- `FRAME_SKIP = 3` — Jeden N-ten Frame analysieren

### Cron (alle 5 Minuten) — Email-Fallback

```bash
crontab -u gh -l
```

Einträge:
```cron
*/5 * * * * /home/gh/python/reolink_AI/run_mail_processor.sh >> .../logs/mail_chain.log 2>&1
```

Funktion: Verarbeitet eingehende Emails von Cam2 als Fallback
(falls WLAN-Stream ausgefallen). Extrahiert MP4/JPG-Anhänge und
triggert ebenfalls die Chain.

## Chain (run_chain.sh)

Wird von beiden Prozessen getriggert:

```
run_chain.sh --base-path /var/www/web2/YYYY/MM --limit N
    │
    ├── Step 1: run_person.sh  → person.py
    │     YOLO + InsightFace (GPU)
    │     Erstellt annotiertes H.264-Video mit Bounding Boxes
    │
    ├── Step 2: run_cluster.sh → cam2_cluster_faces.py
    │     DBSCAN-Clustering (eps=0.4, min_samples=2)
    │
    └── Step 3: run_report.sh  → cam2_report.py
          Statistik-Log
```

Logs:
- `logs/chain.log` — Chain-Gesamtlog
- `logs/person.log` — person.py Detail-Log
- `logs/cam2_stream.log` — Stream-Monitor-Log
- `logs/mail_chain.log` — Email-Fallback-Log

## Apache-Konfiguration

`/etc/apache2/sites-available/cam_stream.conf`

Aliases:
- `/web1` → `/var/www/web1/` (Cam1 FTP + annotierte Bilder)
- `/web2` → `/var/www/web2/` (Cam2 RTSP-Clips)

```bash
sudo apache2ctl configtest
sudo systemctl reload apache2
```

## Postfix

Empfängt Emails von Cam2 (192.168.178.0/24) auf Port 25.
Zustellung in Maildir `/home/gh/Maildir/new/`.

```bash
systemctl status postfix
# Test:
swaks --to gh@heissa.de --server 192.168.5.23
```

## GPU-Environment

Für alle AI-Prozesse (systemd + cron):
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0
export CUDA_MODULE_LOADING=LAZY
```

Gesetzt in: `run_person.sh`, `run_chain.sh`, `cam2-stream.service`
