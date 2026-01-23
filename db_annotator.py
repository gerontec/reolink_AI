#!/home/gh/python/venv_py311/bin/python3
"""
DB-Annotator: Erstellt annotierte Bilder basierend auf watchdog2.py Detektionen aus MariaDB
- Liest neueste Recordings aus cam2_recordings
- Holt Detektionen aus cam2_detected_objects und cam2_detected_faces
- Zeichnet Bounding Boxes auf Original-Bilder
- Speichert annotierte JPGs mit Timestamp
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pymysql
import cv2
import numpy as np

# Konfiguration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

MEDIA_BASE_PATH = Path("/var/www/web1")
ANNOTATED_OUTPUT = Path("/var/www/web1/annotated_from_db")
INTERVAL_SECONDS = 60  # Jede Minute
KEEP_HOURS = 24

# Farben (BGR Format für OpenCV)
COLOR_VEHICLE = (0, 255, 255)   # Gelb
COLOR_PERSON = (0, 255, 0)      # Grün
COLOR_KNOWN_FACE = (0, 255, 255)  # Gelb
COLOR_UNKNOWN_FACE = (0, 0, 255) # Rot

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/db_annotator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Erstellt DB-Verbindung"""
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"DB-Verbindung fehlgeschlagen: {e}")
        return None


def get_latest_recordings(conn, limit: int = 10) -> List[Dict]:
    """
    Holt die neuesten Recordings aus der DB

    Args:
        conn: DB-Connection
        limit: Anzahl Recordings

    Returns:
        Liste von Recording-Dicts
    """
    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        query = """
            SELECT id, camera_name, file_path, file_type, recorded_at, analyzed
            FROM cam2_recordings
            WHERE file_type = 'jpg' AND analyzed = 1
            ORDER BY recorded_at DESC
            LIMIT %s
        """

        cursor.execute(query, (limit,))
        recordings = cursor.fetchall()
        cursor.close()

        return recordings

    except Exception as e:
        logger.error(f"Fehler beim Laden der Recordings: {e}")
        return []


def get_detections_for_recording(conn, recording_id: int) -> Dict[str, Any]:
    """
    Holt alle Detektionen für ein Recording

    Args:
        conn: DB-Connection
        recording_id: Recording ID

    Returns:
        Dict mit 'objects' und 'faces' Listen
    """
    detections = {
        'objects': [],
        'faces': []
    }

    try:
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        # Objekte/Fahrzeuge holen
        query = """
            SELECT object_class, confidence,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, parking_spot_id
            FROM cam2_detected_objects
            WHERE recording_id = %s
        """
        cursor.execute(query, (recording_id,))
        detections['objects'] = cursor.fetchall()

        # Gesichter holen
        query = """
            SELECT person_name, confidence,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2
            FROM cam2_detected_faces
            WHERE recording_id = %s
        """
        cursor.execute(query, (recording_id,))
        detections['faces'] = cursor.fetchall()

        cursor.close()

        return detections

    except Exception as e:
        logger.error(f"Fehler beim Laden der Detektionen: {e}")
        return detections


def draw_bbox(image: np.ndarray, bbox: Dict, color: tuple,
              label: str, thickness: int = 3):
    """Zeichnet Bounding Box mit Label"""
    x1 = int(bbox['bbox_x1'])
    y1 = int(bbox['bbox_y1'])
    x2 = int(bbox['bbox_x2'])
    y2 = int(bbox['bbox_y2'])

    # Box zeichnen
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Label mit Hintergrund
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    # Hintergrund
    cv2.rectangle(image,
                 (x1, y1 - label_size[1] - 10),
                 (x1 + label_size[0], y1),
                 color, -1)

    # Text
    cv2.putText(image, label, (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def annotate_image_from_db(image_path: Path, detections: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Erstellt annotiertes Bild basierend auf DB-Detektionen

    Args:
        image_path: Pfad zum Original-Bild
        detections: Dict mit 'objects' und 'faces'

    Returns:
        Annotiertes Bild als numpy array oder None
    """
    try:
        # Bild laden
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Bild konnte nicht geladen werden: {image_path}")
            return None

        detection_count = 0
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']

        # 1. Fahrzeuge (Gelb)
        for obj in detections['objects']:
            if obj['object_class'] in vehicle_classes:
                label = f"{obj['object_class']} {obj['confidence']:.2f}"
                if obj.get('parking_spot_id'):
                    label += f" P{obj['parking_spot_id']}"
                draw_bbox(image, obj, COLOR_VEHICLE, label)
                detection_count += 1

        # 2. Personen (Grün)
        for obj in detections['objects']:
            if obj['object_class'] == 'person':
                label = f"Person {obj['confidence']:.2f}"
                draw_bbox(image, obj, COLOR_PERSON, label)
                detection_count += 1

        # 3. Gesichter (Gelb = bekannt, Rot = unbekannt)
        for face in detections['faces']:
            is_known = face['person_name'] != 'Unknown'
            color = COLOR_KNOWN_FACE if is_known else COLOR_UNKNOWN_FACE
            label = f"{face['person_name']} {face['confidence']:.2f}"
            thickness = 4 if is_known else 2
            draw_bbox(image, face, color, label)
            detection_count += 1

        if detection_count == 0:
            logger.debug(f"Keine Detektionen in {image_path.name}")
            return None

        # Info-Text
        info_text = f"Detektionen: {detection_count} | DB-basiert"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        return image

    except Exception as e:
        logger.error(f"Fehler beim Annotieren: {e}")
        return None


def cleanup_old_files(directory: Path, hours: int):
    """Löscht Dateien älter als X Stunden"""
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


def process_latest_recordings(conn, limit: int = 5):
    """
    Verarbeitet die neuesten Recordings

    Args:
        conn: DB-Connection
        limit: Anzahl zu verarbeitender Recordings
    """
    # Hole neueste Recordings
    recordings = get_latest_recordings(conn, limit)

    if not recordings:
        logger.info("Keine neuen Recordings gefunden")
        return

    logger.info(f"Verarbeite {len(recordings)} Recording(s)...")

    processed = 0
    for recording in recordings:
        try:
            # Original-Bildpfad
            image_path = MEDIA_BASE_PATH / recording['file_path']

            if not image_path.exists():
                logger.warning(f"Bild nicht gefunden: {image_path}")
                continue

            # Detektionen laden
            detections = get_detections_for_recording(conn, recording['id'])

            if not detections['objects'] and not detections['faces']:
                logger.debug(f"Keine Detektionen für Recording {recording['id']}")
                continue

            # Annotiertes Bild erstellen
            annotated = annotate_image_from_db(image_path, detections)

            if annotated is not None:
                # Speichern
                output_filename = f"db_{recording['camera_name']}_{recording['recorded_at'].strftime('%Y%m%d_%H%M%S')}.jpg"
                output_path = ANNOTATED_OUTPUT / output_filename

                cv2.imwrite(str(output_path), annotated)
                logger.info(f"✓ Gespeichert: {output_filename}")
                processed += 1

        except Exception as e:
            logger.error(f"Fehler bei Recording {recording['id']}: {e}")
            continue

    logger.info(f"✓ {processed} Bilder annotiert")


def main_loop(interval_seconds: int = 60):
    """
    Hauptschleife: Erstellt jede Minute annotierte Bilder aus DB

    Args:
        interval_seconds: Intervall in Sekunden
    """
    logger.info("=" * 70)
    logger.info("DB-Annotator gestartet (watchdog2.py Datenquelle)")
    logger.info(f"Intervall: {interval_seconds} Sekunden")
    logger.info(f"Output: {ANNOTATED_OUTPUT}")
    logger.info(f"Aufbewahrung: {KEEP_HOURS} Stunden")
    logger.info("=" * 70)

    # Verzeichnis erstellen
    ANNOTATED_OUTPUT.mkdir(parents=True, exist_ok=True)

    iteration = 0

    while True:
        try:
            iteration += 1
            logger.info(f"\n[{iteration}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # DB-Verbindung
            conn = get_db_connection()
            if not conn:
                logger.error("Keine DB-Verbindung - überspringe Iteration")
                time.sleep(interval_seconds)
                continue

            # Verarbeite neueste Recordings
            process_latest_recordings(conn, limit=5)

            conn.close()

            # Cleanup (alle 10 Iterationen)
            if iteration % 10 == 0:
                cleanup_old_files(ANNOTATED_OUTPUT, KEEP_HOURS)

            # Warten
            logger.info(f"⏸ Warte {interval_seconds}s...\n")
            time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("\n✓ DB-Annotator gestoppt (Ctrl+C)")
            break
        except Exception as e:
            logger.error(f"✗ Fehler in Hauptschleife: {e}")
            import traceback
            logger.error(traceback.format_exc())
            time.sleep(interval_seconds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='DB-Annotator: Erstellt annotierte Bilder aus watchdog2.py Detektionen'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Intervall in Sekunden (Standard: 60)'
    )
    parser.add_argument(
        '--keep-hours',
        type=int,
        default=24,
        help='Bilder älter als X Stunden löschen (Standard: 24)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Anzahl Recordings pro Iteration (Standard: 5)'
    )

    args = parser.parse_args()

    if args.keep_hours:
        KEEP_HOURS = args.keep_hours

    main_loop(args.interval)
