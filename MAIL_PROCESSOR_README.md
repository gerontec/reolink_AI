# Email Attachment Processor

Fallback-Pipeline für Cam2: Verarbeitet MP4/JPG-Anhänge aus eingehenden
Alarm-Emails wenn der RTSP-Stream nicht verfügbar ist.

## Ablauf

```
Cam2 erkennt Person
    → sendet Email an gh@heissa.de (Postfix, Port 25)
    → landet in /home/gh/Maildir/new/

Cron (alle 5 Min):
    run_mail_processor.sh
        → process_mail_attachments.py
            → extrahiert MP4/JPG-Anhang
            → speichert als Camera2_00_TIMESTAMP.mp4
              nach /var/www/web2/YYYY/MM/
            → Email → /home/gh/python/mail_processed/
        → bei Erfolg: run_chain.sh --base-path ... --limit 50
```

## Dateinamen-Schema

Gespeicherte Dateien: `Camera2_00_YYYYMMDD_HHMMSS.{mp4,jpg}`

Erlaubte Endungen: `.mp4`, `.jpg`, `.jpeg`, `.avi`, `.mov`

## Konfiguration (process_mail_attachments.py)

```python
MAILDIR_NEW  = Path("/home/gh/Maildir/new")
OUTPUT_BASE  = Path("/var/www/web2")
MAILDIR_PROCESSED = Path("/home/gh/python/mail_processed")
```

## Logs

```bash
# Email-Verarbeitung
tail -f /home/gh/python/logs/mail_processor.log

# Chain nach Email-Trigger
tail -f /home/gh/python/reolink_AI/logs/mail_chain.log
```

## Manueller Aufruf

```bash
cd /home/gh/python/reolink_AI
./run_mail_processor.sh
```

## Verhalten bei Fehlern

| Situation | Aktion |
|-----------|--------|
| Anhang gespeichert | Email → mail_processed/, Chain getriggert |
| Anhang gefunden, Speichern fehlgeschlagen | Email → mail_processed/ (kein Chain) |
| Kein erlaubter Anhang | Email gelöscht |

## Hinweis

Der Email-Prozessor ist **Fallback** — primär läuft `cam2-stream.service`
mit RTSP-Erkennung und zeichnet in 1920×1080 auf.
Emails von Cam2 sind 640×360 (Sub-Stream-Qualität).
