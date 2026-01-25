#!/home/gh/python/venv_py311/bin/python3
"""
Email Attachment Processor
Extrahiert Video/Bild-Attachments aus Maildir und kopiert sie zur Analyse
"""

import os
import sys
import email
import logging
from pathlib import Path
from datetime import datetime
from email import policy
from email.parser import BytesParser
import shutil

# Configuration
MAILDIR_NEW = Path("/var/www/web1/Maildir/new")
MAILDIR_PROCESSED = Path("/home/gh/python/mail_processed")  # Alternative: eigenes Verzeichnis
OUTPUT_BASE = Path("/var/www/web1")
ALLOWED_EXTENSIONS = {'.mp4', '.jpg', '.jpeg', '.avi', '.mov'}

# Logging
import logging.handlers

# Setup Syslog + File Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Syslog Handler (für /var/log/mail.log via rsyslog)
syslog_handler = logging.handlers.SysLogHandler(address='/dev/log', facility=logging.handlers.SysLogHandler.LOG_MAIL)
syslog_handler.setFormatter(logging.Formatter('mail_processor[%(process)d]: %(levelname)s - %(message)s'))
logger.addHandler(syslog_handler)

# File Handler (Backup in Home-Verzeichnis)
file_handler = logging.FileHandler('/home/gh/python/logs/mail_processor.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


def ensure_directories():
    """Erstellt notwendige Verzeichnisse"""
    MAILDIR_PROCESSED.mkdir(parents=True, exist_ok=True)
    logger.info(f"Verzeichnisse überprüft: {MAILDIR_PROCESSED}")


def get_output_directory():
    """Erstellt Ausgabe-Verzeichnis nach Schema: YYYY/MM/"""
    now = datetime.now()
    output_dir = OUTPUT_BASE / now.strftime("%Y") / now.strftime("%m")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sanitize_filename(filename):
    """Bereinigt Dateinamen von problematischen Zeichen"""
    # Entferne gefährliche Zeichen
    filename = filename.replace('/', '_').replace('\\', '_')
    filename = filename.replace(' ', '_').replace('..', '_')
    return filename


def process_email_file(email_path):
    """
    Verarbeitet eine Email-Datei und extrahiert Attachments

    Returns:
        int: Anzahl extrahierter Attachments
    """
    try:
        logger.info(f"Verarbeite Email: {email_path.name}")

        # Email laden
        with open(email_path, 'rb') as fp:
            msg = BytesParser(policy=policy.default).parse(fp)

        # Email-Metadaten loggen
        from_addr = msg.get('From', 'Unknown')
        subject = msg.get('Subject', 'No Subject')
        date = msg.get('Date', 'Unknown')

        logger.info(f"  Von: {from_addr}")
        logger.info(f"  Betreff: {subject}")
        logger.info(f"  Datum: {date}")

        attachment_count = 0
        output_dir = get_output_directory()

        # Durchsuche alle Teile der Email
        for part in msg.walk():
            # Nur Attachments verarbeiten
            content_disposition = part.get_content_disposition()

            if content_disposition == 'attachment':
                filename = part.get_filename()

                if not filename:
                    logger.debug(f"  Attachment ohne Dateinamen übersprungen")
                    continue

                # Prüfe Dateiendung
                file_ext = Path(filename).suffix.lower()
                if file_ext not in ALLOWED_EXTENSIONS:
                    logger.debug(f"  Überspringe {filename} (nicht erlaubte Endung: {file_ext})")
                    continue

                # Bereinige Dateinamen
                filename = sanitize_filename(filename)

                # Füge Zeitstempel hinzu um Duplikate zu vermeiden
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_parts = Path(filename).stem
                extension = Path(filename).suffix
                unique_filename = f"Camera1_00_{timestamp}{extension}"

                output_path = output_dir / unique_filename

                # Speichere Attachment
                try:
                    payload = part.get_payload(decode=True)

                    if payload:
                        with open(output_path, 'wb') as f:
                            f.write(payload)

                        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                        logger.info(f"  ✓ Extrahiert: {unique_filename} ({file_size:.2f} MB)")
                        logger.info(f"    Ziel: {output_path}")

                        # Setze Berechtigungen
                        os.chmod(output_path, 0o664)
                        shutil.chown(output_path, user='gh', group='www-data')

                        attachment_count += 1
                    else:
                        logger.warning(f"  Leeres Attachment: {filename}")

                except Exception as e:
                    logger.error(f"  Fehler beim Speichern von {filename}: {e}")
                    continue

        if attachment_count == 0:
            logger.info(f"  Keine Attachments gefunden oder extrahiert")

        return attachment_count

    except Exception as e:
        logger.error(f"Fehler beim Verarbeiten von {email_path.name}: {e}")
        return 0


def move_to_processed(email_path):
    """Verschiebt verarbeitete Email ins processed-Verzeichnis"""
    try:
        # Füge Zeitstempel zum Namen hinzu
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{timestamp}_{email_path.name}"
        dest_path = MAILDIR_PROCESSED / new_name

        shutil.move(str(email_path), str(dest_path))
        logger.info(f"  → Email archiviert: {new_name}")

    except Exception as e:
        logger.error(f"Fehler beim Archivieren von {email_path.name}: {e}")


def process_maildir():
    """
    Verarbeitet alle Emails im Maildir/new Verzeichnis

    Returns:
        tuple: (anzahl_emails, anzahl_attachments)
    """
    if not MAILDIR_NEW.exists():
        logger.error(f"Maildir nicht gefunden: {MAILDIR_NEW}")
        return 0, 0

    email_files = list(MAILDIR_NEW.glob('*'))

    if not email_files:
        logger.info("Keine neuen Emails gefunden")
        return 0, 0

    logger.info(f"Gefunden: {len(email_files)} Email(s)")

    total_emails = 0
    total_attachments = 0

    for email_path in email_files:
        # Überspringe Verzeichnisse und versteckte Dateien
        if email_path.is_dir() or email_path.name.startswith('.'):
            continue

        attachment_count = process_email_file(email_path)
        total_attachments += attachment_count
        total_emails += 1

        # Verschiebe verarbeitete Email
        move_to_processed(email_path)

    return total_emails, total_attachments


def main():
    """Hauptfunktion"""
    logger.info("=" * 70)
    logger.info("Email Attachment Processor gestartet")
    logger.info("=" * 70)

    # Erstelle Verzeichnisse
    ensure_directories()

    # Verarbeite Emails
    emails_processed, attachments_extracted = process_maildir()

    logger.info("=" * 70)
    logger.info(f"Verarbeitung abgeschlossen")
    logger.info(f"  Emails verarbeitet: {emails_processed}")
    logger.info(f"  Attachments extrahiert: {attachments_extracted}")
    logger.info("=" * 70)

    return 0 if emails_processed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
