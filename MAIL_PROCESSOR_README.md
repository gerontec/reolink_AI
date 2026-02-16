# Email Attachment Processor

Verarbeitet Video/Bild-Attachments aus Maildir und kopiert sie zur AI-Analyse.

## Installation

### 1. Dateien kopieren

```bash
cd ~/python/reolink_AI

# Script ausführbar machen
chmod +x process_mail_attachments.py
chmod +x run_mail_processor.sh

# Logs-Verzeichnis erstellen
mkdir -p /home/gh/python/logs
```

### 2. Berechtigungen setzen

```bash
# User gh muss Maildir lesen können
sudo usermod -a -G web1 gh

# Neu anmelden damit Gruppe aktiv wird
# ODER: newgrp web1
```

### 3. Test-Lauf

```bash
# Manueller Test
./process_mail_attachments.py

# ODER via Wrapper-Script
./run_mail_processor.sh
```

## Cron-Job einrichten

### Variante 1: Alle 5 Minuten

```bash
# Crontab bearbeiten
crontab -e

# Folgende Zeile hinzufügen:
*/5 * * * * /home/gh/python/reolink_AI/run_mail_processor.sh >> /home/gh/python/logs/mail_processor_cron.log 2>&1
```

### Variante 2: Alle 10 Minuten

```bash
*/10 * * * * /home/gh/python/reolink_AI/run_mail_processor.sh >> /home/gh/python/logs/mail_processor_cron.log 2>&1
```

### Variante 3: Alle 30 Minuten

```bash
*/30 * * * * /home/gh/python/reolink_AI/run_mail_processor.sh >> /home/gh/python/logs/mail_processor_cron.log 2>&1
```

## Workflow

```
1. Kamera sendet Email mit Video/Bild-Attachment
   ↓
2. Email landet in /var/www/web1/Maildir/new/
   ↓
3. Cron-Job führt process_mail_attachments.py aus
   ↓
4. Script extrahiert Attachments
   ↓
5. Dateien werden nach /var/www/web1/YYYY/MM/ kopiert
   ↓
6. Email wird nach /var/www/web1/Maildir/.processed/ verschoben
   ↓
7. watchdog2.py analysiert neue Dateien automatisch
```

## Verzeichnisstruktur

```
/var/www/web1/
├── Maildir/
│   ├── new/                    # Neue Emails (werden verarbeitet)
│   └── .processed/             # Archiv verarbeiteter Emails
├── 2026/
│   └── 01/
│       ├── Camera1_00_20260125173000.mp4
│       └── Camera1_00_20260125173500.jpg
└── annotated/                  # YOLO-Annotationen
```

## Logs

```bash
# Haupt-Log
tail -f /home/gh/python/logs/mail_processor.log

# Cron-Job-Log
tail -f /home/gh/python/logs/mail_processor_cron.log

# Watchdog-Log
tail -f /home/gh/python/logs/watchdog.log
```

## Konfiguration

Passe Einstellungen in `process_mail_attachments.py` an:

```python
MAILDIR_NEW = Path("/var/www/web1/Maildir/new")
OUTPUT_BASE = Path("/var/www/web1")
ALLOWED_EXTENSIONS = {'.mp4', '.jpg', '.jpeg', '.avi', '.mov'}
```

## Troubleshooting

### Berechtigungs-Fehler

```bash
# Prüfe ob User gh in Gruppe web1 ist
groups gh

# Falls nicht:
sudo usermod -a -G web1 gh
newgrp web1
```

### Test-Email erstellen

```bash
# Testdatei erstellen
echo -e "From: test@example.com\nSubject: Test\n\nTest-Email" > /var/www/web1/Maildir/new/test.eml

# Verarbeiten
./process_mail_attachments.py
```

### Email-Format prüfen

```bash
# Zeige Email-Struktur
sudo cat /var/www/web1/Maildir/new/1769358588.V10306I3480435M329128.pve
```

## Kombination mit watchdog2.py

Der Mail-Processor läuft **VOR** watchdog2.py:

```bash
# Cron-Jobs (Beispiel):
*/5 * * * * /home/gh/python/reolink_AI/run_mail_processor.sh
*/10 * * * * /home/gh/python/run_watchdog.sh
```

So wird sichergestellt dass:
1. Erst Attachments extrahiert werden (alle 5 Min)
2. Dann AI-Analyse läuft (alle 10 Min)
