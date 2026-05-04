# Reolink AI — Überwachungssystem

KI-gestütztes Überwachungssystem für Reolink-Kameras mit Personenerkennung,
Gesichtserkennung und automatischer Clip-Aufnahme.

## Kameras

| Kamera | Typ | Anbindung | Auflösung |
|--------|-----|-----------|-----------|
| Cam1 | Reolink (FTP) | FTP → `/var/www/web1/` | 2560×1440 JPG |
| Cam2 | Reolink E1 (192.168.178.128) | RTSP-Stream | 1920×1080 MP4 |

## Architektur

```
Cam1 (FTP)                         Cam2 (RTSP)
    │                                   │
    ▼                                   ▼
/var/www/web1/YYYY/MM/          cam2_stream.py (systemd)
Camera1_00_*.jpg                    │
    │                        Sub-Stream 640×360
    │                        YOLO @ jeden 3. Frame
    │                        Person erkannt?
    │                               │ JA (Cooldown 90s)
    │                               ▼
    │                    ffmpeg → Main-Stream 1920×1080
    │                    Camera2_00_TIMESTAMP.mp4 (25s)
    │                    /var/www/web2/YYYY/MM/
    │                               │
    └───────────────────────────────┘
                    │
                    ▼
            run_chain.sh
        ┌───────────────────┐
        │ 1. person.py      │  YOLO + InsightFace (GPU)
        │    sample_rate=10 │  det_thresh=0.3
        │    → annotiertes  │  → H.264 MP4 mit Bboxes
        │      Video (MP4)  │
        ├───────────────────┤
        │ 2. clustering.py  │  DBSCAN eps=0.4
        │    → Cluster      │
        ├───────────────────┤
        │ 3. report.py      │  Statistik
        └───────────────────┘
                    │
                    ▼
            MariaDB wagodb
        cam2_recordings
        cam2_detected_faces
        cam2_detected_objects

                    │
                    ▼
        http://192.168.5.23/
        Gesichter benennen, Cluster anzeigen,
        Video-Player mit Bounding Boxes
```

## Services & Cron

| Prozess | Typ | Intervall |
|---------|-----|-----------|
| `cam2-stream.service` | systemd (dauerhaft) | — |
| `run_mail_processor.sh` | cron | alle 5 Min (Email-Fallback) |

## Verzeichnisse

| Pfad | Inhalt |
|------|--------|
| `/var/www/web1/` | Cam1 FTP-Uploads (JPG) |
| `/var/www/web1/annotated/` | Annotierte Videos/Bilder (H.264 MP4 / JPG) |
| `/var/www/web2/YYYY/MM/` | Cam2 RTSP-Clips (MP4, Camera2_00_*) |
| `/home/gh/python/mail_processed/` | Archivierte Emails |
| `/home/gh/python/reolink_AI/logs/` | Alle Logs |

## Web-Interface

`http://192.168.5.23/` — Admin-Seite

- Zeigt 1 bestes Gesicht pro Cluster (Unknown)
- Video-Player mit eingebetteten Bounding Boxes
- Gesicht benennen → ganzen Cluster umbenennen
- Cluster-Übersicht mit allen Gesichtern
- Filter: Datum, „Auch benannte anzeigen"

## Wichtige Befehle

```bash
# cam2-Stream Status
sudo systemctl status cam2-stream.service
journalctl -u cam2-stream -f

# Chain manuell starten (cam2)
./run_chain.sh --base-path /var/www/web2/$(date +%Y/%m) --limit 10

# Chain manuell starten (cam1)
./run_chain.sh --base-path /var/www/web1/$(date +%Y/%m) --limit 50

# Clustering neu berechnen
./run_cluster.sh

# Datei neu analysieren (force)
./run_person.sh --base-path /var/www/web2/2026/05 --force

# GPU-Status
nvidia-smi
```

## Datenbank

```
wagodb @ localhost
User: gh / a12345

Tabellen:
  cam2_recordings        — Alle Aufnahmen (JPG + MP4)
  cam2_detected_faces    — Erkannte Gesichter + Embeddings
  cam2_detected_objects  — YOLO-Objekte (person, car, ...)
  cam2_analysis_summary  — Pro-Aufnahme Zusammenfassung
```

## Hardware

| Komponente | Details |
|------------|---------|
| Server | Dell (dell-3660), Ubuntu 24.04 |
| GPU | NVIDIA Tesla P4, 8 GB, CUDA 11.8 |
| Python | 3.11, venv: `/home/gh/python/venv_py311/` |
| YOLO | YOLOv8m @ `/opt/models/yolov8m.pt` |
| Face | InsightFace buffalo_s (RetinaFace + ArcFace) |
