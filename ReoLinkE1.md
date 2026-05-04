# Reolink E1 — Cam2 Konfiguration & Architektur

## Hardware

| Eigenschaft | Wert |
|-------------|------|
| Modell | Reolink E1 |
| Name | Front door |
| IP | 192.168.178.128 (WiFi, Netzwerk 192.168.178.0/24) |
| Credentials | admin / 2einfach |
| Offene Ports | 554 (RTSP), 8000 (ONVIF/gSOAP), 9000 (Reolink proprietär) |

## RTSP-Streams

| Stream | URL | Auflösung | FPS | Bitrate |
|--------|-----|-----------|-----|---------|
| Main | `rtsp://admin:2einfach@192.168.178.128:554/h264Preview_01_main` | 1920×1080 | 10fps | ~1.1 Mbit/s |
| Sub  | `rtsp://admin:2einfach@192.168.178.128:554/h264Preview_01_sub`  | 640×360   | 25fps | ~200 kbit/s |

## WiFi-Stabilitätsmessungen (2026-05-03)

| Konfiguration | Dauer | Fehler |
|---------------|-------|--------|
| Main 2560×1440 @ 30fps | ~28s | CSeq-Drop |
| Main 2560×1440 @ 10fps | ~12s | CSeq-Drop |
| Main 1920×1080 @ 15fps | 25s ✅ | 0 |
| Main 1920×1080 @ 10fps | 25s ✅ | 0 |
| Sub  640×360   @ 25fps | 6 min ✅ | 0 |

**Fazit:** 2560×1440 übersteigt die WiFi-Kapazität. 1920×1080 ist stabil.

## Verarbeitungs-Architektur

```
cam2_stream.py (systemd: cam2-stream.service, dauerhaft)
    │
    ├── Sub-Stream 640×360 @ 25fps (TCP)
    │   └── YOLO jeden 3. Frame (classes=[person], conf>0.45)
    │       └── Person erkannt?
    │           ├── NEIN → Frame verwerfen, kein Speichern
    │           └── JA  → trigger_clip() [Cooldown: 90s]
    │
    └── Clip-Thread (bei Trigger, non-blocking)
        ├── ffmpeg Main-Stream → /var/www/web2/YYYY/MM/Camera2_00_TIMESTAMP.mp4
        ├── Clip < 50KB → verwerfen
        └── Clip OK → run_chain.sh --base-path /var/www/web2/YYYY/MM --limit 5
                        ├── person.py  (YOLO + InsightFace)
                        │   ├── sample_rate=10 (jeden 10. Frame)
                        │   ├── det_thresh=0.3 (kleine Gesichter)
                        │   └── → annotiertes H.264-Video mit Bboxes
                        ├── cam2_cluster_faces.py (DBSCAN eps=0.4)
                        └── cam2_report.py
```

## Dateipfade

| Zweck | Pfad |
|-------|------|
| Rohe Clips (MP4) | `/var/www/web2/YYYY/MM/Camera2_00_YYYYMMDD_HHMMSS.mp4` |
| Annotierte Videos | `/var/www/web1/annotated/video_Camera2_00_*.mp4` (H.264) |
| Stream-Log | `/home/gh/python/reolink_AI/logs/cam2_stream.log` |
| Chain-Log | `/home/gh/python/reolink_AI/logs/chain.log` |

## Web-URLs

| Ressource | URL |
|-----------|-----|
| Admin-Interface | `http://192.168.5.23/` |
| Rohe Clips | `http://192.168.5.23/web2/YYYY/MM/Camera2_00_*.mp4` |
| Annotierte Videos | `http://192.168.5.23/web1/annotated/video_Camera2_00_*.mp4` |

## InsightFace-Parameter (person.py)

| Parameter | Wert | Begründung |
|-----------|------|------------|
| `det_thresh` | 0.3 | Kleine Gesichter (1920×1080, kurze Clips) |
| `sample_rate` | 10 | Jeden 10. Frame → bei 10fps = jede Sekunde |
| `det_size` | (640,640) | RetinaFace Eingabegröße |
| Model | buffalo_s | RetinaFace + ArcFace, GPU |

## Service-Befehle

```bash
sudo systemctl start   cam2-stream.service
sudo systemctl stop    cam2-stream.service
sudo systemctl restart cam2-stream.service
sudo systemctl status  cam2-stream.service
journalctl -u cam2-stream -f
tail -f /home/gh/python/reolink_AI/logs/cam2_stream.log
```

## Email-Fallback

Die alte Email-Pipeline ist weiterhin als Fallback aktiv:
- Cam2 sendet Alarm-Email mit MP4-Anhang (640×360, Sub-Stream-Qualität)
- Postfix empfängt auf 192.168.5.23:25 von 192.168.178.0/24
- Cron alle 5 Min: `run_mail_processor.sh` → extrahiert → Chain

## Annotiertes Video

`person.py` erstellt für jeden Clip ein annotiertes H.264-Video:
- YOLO-Bboxes (grün=Person, gelb=Fahrzeug)
- InsightFace-Bboxes (rot=Unbekannt, gelb=Bekannt)
- Bboxes auf allen Frames (interpoliert von jedem 10. Frame)
- Browser-kompatibel (libx264, faststart)
- Anzeige als `<video>`-Tag in index.php

## Bekannte Einschränkungen

- Main-Stream WiFi: stabiler Dauerbetrieb nicht möglich → nur kurze Clips bei Bedarf
- Gesichter in 640×360 Sub-Stream: 16–52px → Clustering erst nach mehreren Treffern
- Annotierte Bboxes auf nicht analysierten Frames: letzte bekannte Position (interpoliert)
