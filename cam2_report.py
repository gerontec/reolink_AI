#!/usr/bin/env python3
"""
CAM2 Report - Comprehensive Database Statistics Report
Analyzes all cam2_ tables and generates detailed statistics
"""

import pymysql
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys

DB_CONFIG = {
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

def connect_db():
    """Verbindet mit der Datenbank"""
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Datenbankverbindung fehlgeschlagen: {e}")
        sys.exit(1)

def print_header(title: str, width: int = 80):
    """Druckt formatierten Header"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def print_section(title: str, width: int = 80):
    """Druckt Section-Header"""
    print(f"\n{title}")
    print("-" * width)

def format_size(bytes: int) -> str:
    """Formatiert Dateigröße"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

def get_recordings_stats(conn) -> Dict[str, Any]:
    """Statistiken zu Recordings"""
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    stats = {}

    # Gesamt-Statistiken
    cursor.execute("""
        SELECT
            COUNT(*) as total_recordings,
            SUM(CASE WHEN file_type = 'mp4' THEN 1 ELSE 0 END) as videos,
            SUM(CASE WHEN file_type = 'jpg' THEN 1 ELSE 0 END) as images,
            SUM(file_size) as total_size,
            SUM(CASE WHEN analyzed = 1 THEN 1 ELSE 0 END) as analyzed,
            MIN(recorded_at) as first_recording,
            MAX(recorded_at) as last_recording
        FROM cam2_recordings
    """)
    stats['general'] = cursor.fetchone()

    # Recordings pro Tag (letzte 7 Tage)
    cursor.execute("""
        SELECT
            DATE(recorded_at) as date,
            COUNT(*) as count,
            SUM(CASE WHEN file_type = 'mp4' THEN 1 ELSE 0 END) as videos
        FROM cam2_recordings
        WHERE recorded_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY DATE(recorded_at)
        ORDER BY date DESC
        LIMIT 7
    """)
    stats['daily'] = cursor.fetchall()

    # Recordings pro Stunde (heute)
    cursor.execute("""
        SELECT
            HOUR(recorded_at) as hour,
            COUNT(*) as count
        FROM cam2_recordings
        WHERE DATE(recorded_at) = CURDATE()
        GROUP BY HOUR(recorded_at)
        ORDER BY hour
    """)
    stats['hourly'] = cursor.fetchall()

    cursor.close()
    return stats

def get_faces_stats(conn) -> Dict[str, Any]:
    """Statistiken zu Gesichtern"""
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    stats = {}

    # Gesamt-Statistiken
    cursor.execute("""
        SELECT
            COUNT(*) as total_faces,
            COUNT(DISTINCT person_name) as unique_persons,
            SUM(CASE WHEN person_name = 'Unknown' THEN 1 ELSE 0 END) as unknown_faces,
            AVG(confidence) as avg_confidence
        FROM cam2_detected_faces
    """)
    stats['general'] = cursor.fetchone()

    # Top Personen
    cursor.execute("""
        SELECT
            person_name,
            COUNT(*) as detections,
            AVG(confidence) as avg_confidence
        FROM cam2_detected_faces
        WHERE person_name != 'Unknown'
        GROUP BY person_name
        ORDER BY detections DESC
        LIMIT 10
    """)
    stats['top_persons'] = cursor.fetchall()

    # Neue Gesichter (letzte 24h)
    cursor.execute("""
        SELECT COUNT(*) as new_faces
        FROM cam2_detected_faces
        WHERE detected_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
    """)
    stats['new_24h'] = cursor.fetchone()['new_faces']

    cursor.close()
    return stats

def get_objects_stats(conn) -> Dict[str, Any]:
    """Statistiken zu erkannten Objekten"""
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    stats = {}

    # Objekt-Verteilung
    cursor.execute("""
        SELECT
            object_class,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence
        FROM cam2_detected_objects
        GROUP BY object_class
        ORDER BY count DESC
        LIMIT 15
    """)
    stats['distribution'] = cursor.fetchall()

    # Fahrzeug-Statistiken
    cursor.execute("""
        SELECT
            object_class,
            COUNT(*) as count,
            COUNT(DISTINCT parking_spot_id) as unique_spots
        FROM cam2_detected_objects
        WHERE object_class IN ('car', 'truck', 'bus', 'motorcycle', 'bicycle')
        GROUP BY object_class
        ORDER BY count DESC
    """)
    stats['vehicles'] = cursor.fetchall()

    cursor.close()
    return stats

def get_parking_stats(conn) -> Dict[str, Any]:
    """Statistiken zu Parkplätzen"""
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    stats = {}

    # Parkplatz-Auslastung
    cursor.execute("""
        SELECT
            parking_spot_id,
            COUNT(*) as occupancy_count,
            COUNT(DISTINCT DATE(detected_at)) as days_used
        FROM cam2_detected_objects
        WHERE parking_spot_id IS NOT NULL
        GROUP BY parking_spot_id
        ORDER BY parking_spot_id
    """)
    stats['occupancy'] = cursor.fetchall()

    # Meistgenutzte Parkplätze
    cursor.execute("""
        SELECT
            parking_spot_id,
            COUNT(*) as usage_count,
            GROUP_CONCAT(DISTINCT object_class) as vehicle_types
        FROM cam2_detected_objects
        WHERE parking_spot_id IS NOT NULL
        GROUP BY parking_spot_id
        ORDER BY usage_count DESC
        LIMIT 5
    """)
    stats['top_spots'] = cursor.fetchall()

    # Parkplatz-Nutzung nach Stunde
    cursor.execute("""
        SELECT
            HOUR(o.detected_at) as hour,
            COUNT(DISTINCT parking_spot_id) as active_spots,
            COUNT(*) as total_vehicles
        FROM cam2_detected_objects o
        WHERE parking_spot_id IS NOT NULL
          AND o.detected_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
        GROUP BY HOUR(o.detected_at)
        ORDER BY hour
    """)
    stats['hourly'] = cursor.fetchall()

    cursor.close()
    return stats

def get_scene_stats(conn) -> Dict[str, Any]:
    """Statistiken zu Szenen"""
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    stats = {}

    # Szenen-Verteilung
    cursor.execute("""
        SELECT
            scene_category,
            COUNT(*) as count,
            AVG(total_vehicles) as avg_vehicles,
            AVG(max_persons) as avg_persons
        FROM cam2_analysis_summary
        GROUP BY scene_category
        ORDER BY count DESC
    """)
    stats['distribution'] = cursor.fetchall()

    # GPU-Nutzung
    cursor.execute("""
        SELECT
            SUM(CASE WHEN gpu_used = 1 THEN 1 ELSE 0 END) as gpu_analyses,
            COUNT(*) as total_analyses
        FROM cam2_analysis_summary
    """)
    stats['gpu'] = cursor.fetchone()

    cursor.close()
    return stats

def get_last_annotated_frame(conn) -> Dict[str, Any]:
    """Holt Datum/Uhrzeit des letzten annotierten Frames"""
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("""
        SELECT
            recorded_at,
            file_path,
            annotated_image_path
        FROM cam2_recordings
        WHERE annotated_image_path IS NOT NULL
        ORDER BY recorded_at DESC
        LIMIT 1
    """)

    result = cursor.fetchone()
    cursor.close()
    return result

def print_recordings_report(stats: Dict[str, Any]):
    """Druckt Recordings-Report"""
    print_section("📹 RECORDINGS")

    gen = stats['general']
    print(f"Gesamt Recordings:     {gen['total_recordings']:,}")
    print(f"  - Videos (MP4):      {gen['videos']:,}")
    print(f"  - Bilder (JPG):      {gen['images']:,}")
    print(f"  - Analysiert:        {gen['analyzed']:,} ({gen['analyzed']/gen['total_recordings']*100:.1f}%)")
    print(f"Gesamtgröße:          {format_size(gen['total_size'] or 0)}")
    print(f"Zeitraum:             {gen['first_recording']} bis {gen['last_recording']}")

    if stats['daily']:
        print(f"\n📅 Letzte 7 Tage:")
        for day in stats['daily']:
            print(f"  {day['date']}: {day['count']:3} Dateien ({day['videos']:3} Videos)")

    if stats['hourly']:
        print(f"\n🕐 Heute nach Stunden:")
        for hour in stats['hourly']:
            bar = "█" * (hour['count'] // 2)
            print(f"  {hour['hour']:02d}:00 - {bar} {hour['count']}")

def print_faces_report(stats: Dict[str, Any]):
    """Druckt Gesichter-Report"""
    print_section("👤 GESICHTSERKENNUNG")

    gen = stats['general']
    print(f"Gesamt Gesichter:      {gen['total_faces']:,}")
    print(f"  - Bekannt:           {gen['unique_persons'] - 1:,} Personen")  # -1 für Unknown
    print(f"  - Unbekannt:         {gen['unknown_faces']:,}")
    print(f"Durchschn. Konfidenz:  {gen['avg_confidence']:.2%}")
    print(f"Neue Gesichter (24h):  {stats['new_24h']:,}")

    if stats['top_persons']:
        print(f"\n🏆 Top Erkannte Personen:")
        for i, person in enumerate(stats['top_persons'], 1):
            print(f"  {i:2}. {person['person_name']:20} - {person['detections']:4} Erkennungen ({person['avg_confidence']:.2%})")

def print_objects_report(stats: Dict[str, Any]):
    """Druckt Objekt-Report"""
    print_section("🎯 OBJEKT-ERKENNUNG")

    if stats['distribution']:
        print("Erkannte Objekte:")
        for obj in stats['distribution']:
            bar = "█" * (obj['count'] // 50 if obj['count'] > 50 else 1)
            print(f"  {obj['object_class']:15} {bar:30} {obj['count']:6} ({obj['avg_confidence']:.1%})")

    if stats['vehicles']:
        print(f"\n🚗 Fahrzeug-Statistik:")
        for veh in stats['vehicles']:
            spots = f"auf {veh['unique_spots']} Parkplätzen" if veh['unique_spots'] else ""
            print(f"  {veh['object_class']:15} {veh['count']:6} {spots}")

def print_parking_report(stats: Dict[str, Any]):
    """Druckt Parkplatz-Report"""
    print_section("🅿️  PARKPLATZ-AUSLASTUNG")

    if stats['occupancy']:
        print("Parkplatz-Nutzung (Grid 4x3):")
        print("  Spot  | Belegungen | Genutzte Tage")
        print("  ------|------------|---------------")
        for spot in stats['occupancy']:
            spot_id = spot['parking_spot_id']
            # Berechne Grid-Position
            row = (spot_id - 1) // 4
            col = (spot_id - 1) % 4
            grid_pos = f"R{row+1}C{col+1}"
            print(f"  #{spot_id:2} {grid_pos} | {spot['occupancy_count']:10,} | {spot['days_used']:13}")

    if stats['top_spots']:
        print(f"\n🏆 Meistgenutzte Parkplätze:")
        for i, spot in enumerate(stats['top_spots'], 1):
            print(f"  {i}. Parkplatz #{spot['parking_spot_id']:2} - {spot['usage_count']:6,} Belegungen ({spot['vehicle_types']})")

    if stats['hourly']:
        print(f"\n🕐 Auslastung nach Uhrzeit (letzte 7 Tage):")
        max_spots = max((h['active_spots'] for h in stats['hourly']), default=0)
        for hour in stats['hourly']:
            bar = "█" * int(hour['active_spots'] / max_spots * 30) if max_spots > 0 else ""
            print(f"  {hour['hour']:02d}:00 {bar:30} {hour['active_spots']:2} Plätze, {hour['total_vehicles']:3} Fahrzeuge")

def print_scene_report(stats: Dict[str, Any]):
    """Druckt Szenen-Report"""
    print_section("🎬 SZENEN-KLASSIFIKATION")

    if stats['distribution']:
        print("Szenen-Verteilung:")
        total = sum(s['count'] for s in stats['distribution'])
        for scene in stats['distribution']:
            pct = scene['count'] / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 2)
            print(f"  {scene['scene_category']:12} {bar:30} {scene['count']:6} ({pct:5.1f}%)")
            print(f"               ø {scene['avg_vehicles']:.1f} Fahrzeuge, {scene['avg_persons']:.1f} Personen")

    gpu = stats['gpu']
    if gpu:
        gpu_pct = gpu['gpu_analyses'] / gpu['total_analyses'] * 100 if gpu['total_analyses'] > 0 else 0
        print(f"\n🖥️  GPU-Beschleunigung:   {gpu['gpu_analyses']:,} / {gpu['total_analyses']:,} ({gpu_pct:.1f}%)")

def main():
    """Hauptfunktion"""
    print_header("CAM2 DATABASE REPORT", 80)
    print(f"Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    conn = connect_db()

    try:
        # Recordings
        recordings_stats = get_recordings_stats(conn)
        print_recordings_report(recordings_stats)

        # Gesichter
        faces_stats = get_faces_stats(conn)
        print_faces_report(faces_stats)

        # Objekte
        objects_stats = get_objects_stats(conn)
        print_objects_report(objects_stats)

        # Parkplätze
        parking_stats = get_parking_stats(conn)
        print_parking_report(parking_stats)

        # Szenen
        scene_stats = get_scene_stats(conn)
        print_scene_report(scene_stats)

        # Letzter annotierter Frame
        last_annotated = get_last_annotated_frame(conn)
        if last_annotated:
            print(f"\n📸 Letzter annotierter Frame: {last_annotated['recorded_at']} ({last_annotated['file_path']})")

        print("\n" + "=" * 80)
        print("✓ Report erfolgreich generiert")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Fehler beim Generieren des Reports: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
