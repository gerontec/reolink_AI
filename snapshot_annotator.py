#!/home/gh/python/venv_py311/bin/python3
"""
Snapshot-Annotator: Erstellt jede Minute ein annotiertes Bild mit YOLO-Detektionen
- Macht Snapshot von Reolink-Kamera
- Führt YOLO-Analyse durch
- Zeichnet Bounding Boxes (Fahrzeuge, Personen, Gesichter)
- Speichert annotiertes Bild mit Timestamp
"""

import os
import sys
import time
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import requests
from requests.auth import HTTPDigestAuth

# AI Imports
os.environ['ORT_DISABLE_CUDNN_FRONTEND'] = '1'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Importiere AI-Analyzer und ImageAnnotator aus watchdog2.py
sys.path.insert(0, str(Path(__file__).parent))
from watchdog2 import AIAnalyzer, ImageAnnotator

# Konfiguration
CAMERA_IP = "192.168.178.xxx"  # ANPASSEN!
CAMERA_USER = "admin"
CAMERA_PASS = "password"  # ANPASSEN!
CAMERA_NAME = "Camera1"

SNAPSHOT_DIR = Path("/var/www/web1/snapshots")
ANNOTATED_DIR = Path("/var/www/web1/snapshots_annotated")
YOLO_MODEL = "/opt/models/yolov8l.pt"
KNOWN_FACES_DIR = "/opt/known_faces"

# Aufbewahrungszeit (in Stunden)
KEEP_HOURS = 24

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/snapshot_annotator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_camera_snapshot(output_path: Path) -> bool:
    """
    Holt Snapshot von Reolink-Kamera via HTTP

    Returns:
        True bei Erfolg, False bei Fehler
    """
    try:
        # Reolink Snapshot URL
        url = f"http://{CAMERA_IP}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=snapshot"

        # HTTP Digest Auth (Reolink verwendet Digest statt Basic)
        response = requests.get(
            url,
            auth=HTTPDigestAuth(CAMERA_USER, CAMERA_PASS),
            timeout=10
        )

        if response.status_code == 200:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"✓ Snapshot gespeichert: {output_path.name}")
            return True
        else:
            logger.error(f"✗ Snapshot fehlgeschlagen: HTTP {response.status_code}")
            return False

    except Exception as e:
        logger.error(f"✗ Snapshot-Fehler: {e}")
        return False


def cleanup_old_files(directory: Path, hours: int):
    """
    Löscht Dateien älter als X Stunden

    Args:
        directory: Verzeichnis
        hours: Maximale Alter in Stunden
    """
    try:
        now = time.time()
        max_age = hours * 3600

        deleted = 0
        for file in directory.glob("*.jpg"):
            file_age = now - file.stat().st_mtime
            if file_age > max_age:
                file.unlink()
                deleted += 1

        if deleted > 0:
            logger.info(f"✓ {deleted} alte Dateien gelöscht (älter als {hours}h)")

    except Exception as e:
        logger.error(f"✗ Cleanup-Fehler: {e}")


def annotate_snapshot(ai_analyzer: AIAnalyzer, annotator: ImageAnnotator,
                     snapshot_path: Path) -> bool:
    """
    Analysiert Snapshot und erstellt annotiertes Bild

    Returns:
        True bei Erfolg
    """
    try:
        # YOLO-Analyse
        logger.info(f"🔍 Analysiere {snapshot_path.name}...")
        analysis = ai_analyzer.analyze_image(snapshot_path)

        # Statistiken
        vehicles = len(analysis.get('vehicles', []))
        persons = analysis.get('persons', 0)
        faces = len(analysis.get('faces', []))
        known_faces = analysis.get('known_faces_count', 0)

        logger.info(f"  Erkannt: {vehicles} Fahrzeuge, {persons} Personen, "
                   f"{faces} Gesichter ({known_faces} bekannt)")

        # Annotiertes Bild erstellen
        annotated_path = annotator.annotate_image(
            snapshot_path,
            analysis,
            save_prefix=f"annotated_{CAMERA_NAME}"
        )

        if annotated_path:
            logger.info(f"✓ Annotiert: {annotated_path.name}")
            return True
        else:
            logger.warning("⚠ Keine Detektionen - kein annotiertes Bild erstellt")
            return False

    except Exception as e:
        logger.error(f"✗ Annotations-Fehler: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False


def main_loop(interval_seconds: int = 60):
    """
    Hauptschleife: Erstellt jede Minute ein annotiertes Snapshot

    Args:
        interval_seconds: Intervall in Sekunden (Standard: 60 = 1 Minute)
    """
    logger.info("=" * 70)
    logger.info("Snapshot-Annotator gestartet")
    logger.info(f"Kamera: {CAMERA_IP}")
    logger.info(f"Intervall: {interval_seconds} Sekunden")
    logger.info(f"Snapshots: {SNAPSHOT_DIR}")
    logger.info(f"Annotiert: {ANNOTATED_DIR}")
    logger.info(f"Aufbewahrung: {KEEP_HOURS} Stunden")
    logger.info("=" * 70)

    # AI-Analyzer initialisieren
    logger.info("Initialisiere AI-Analyzer...")
    ai_analyzer = AIAnalyzer(
        YOLO_MODEL,
        KNOWN_FACES_DIR,
        force_gpu=True
    )

    # Image-Annotator initialisieren
    annotator = ImageAnnotator(ANNOTATED_DIR)

    # Verzeichnisse erstellen
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("✓ Bereit - starte Snapshot-Loop\n")

    iteration = 0

    while True:
        try:
            iteration += 1
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            logger.info(f"[{iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # 1. Snapshot holen
            snapshot_path = SNAPSHOT_DIR / f"{CAMERA_NAME}_{timestamp}.jpg"
            if not get_camera_snapshot(snapshot_path):
                logger.error("Snapshot fehlgeschlagen - überspringe diese Iteration")
                time.sleep(interval_seconds)
                continue

            # 2. Analysieren und annotieren
            annotate_snapshot(ai_analyzer, annotator, snapshot_path)

            # 3. Alte Dateien löschen (alle 10 Iterationen)
            if iteration % 10 == 0:
                cleanup_old_files(SNAPSHOT_DIR, KEEP_HOURS)
                cleanup_old_files(ANNOTATED_DIR, KEEP_HOURS)

            # 4. Warten bis nächste Minute
            logger.info(f"⏸ Warte {interval_seconds}s bis nächster Snapshot...\n")
            time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\n✓ Snapshot-Annotator gestoppt (Ctrl+C)")
            break
        except Exception as e:
            logger.error(f"✗ Fehler in Hauptschleife: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Snapshot-Annotator: Erstellt jede Minute annotierte Bilder mit YOLO-Detektionen'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Intervall in Sekunden (Standard: 60)'
    )
    parser.add_argument(
        '--camera-ip',
        help='Kamera IP-Adresse (überschreibt Standard)'
    )
    parser.add_argument(
        '--keep-hours',
        type=int,
        default=24,
        help='Bilder älter als X Stunden löschen (Standard: 24)'
    )

    args = parser.parse_args()

    if args.camera_ip:
        CAMERA_IP = args.camera_ip

    if args.keep_hours:
        KEEP_HOURS = args.keep_hours

    # Prüfe ob Kamera-IP konfiguriert ist
    if "xxx" in CAMERA_IP:
        logger.error("✗ FEHLER: Bitte CAMERA_IP im Script anpassen!")
        logger.error("   Zeile 33: CAMERA_IP = '192.168.178.xxx'")
        sys.exit(1)

    main_loop(args.interval)
