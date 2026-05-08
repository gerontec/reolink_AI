#!/usr/bin/env python3
"""
CAM Production Tables - Confidence Score Report
Vergleicht die Confidence Scores zwischen cam_* (Production) und cam2_* (Test)
"""

import pymysql
from datetime import datetime
import sys

DB_CONFIG = {
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

def connect_db():
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Datenbankverbindung fehlgeschlagen: {e}")
        sys.exit(1)

def format_size(bytes):
    """Formatiert Dateigröße"""
    if bytes is None:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024
    return f"{bytes:.2f} TB"

def print_header(title, width=80):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)

def print_section(title, width=80):
    print(f"\n{title}")
    print("-" * width)

def check_confidence_scores(conn, table_prefix):
    """Prüft Confidence Scores für ein Table-Schema"""
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # Objekt-Confidence
    print_section(f"🎯 {table_prefix.upper()} - OBJEKT CONFIDENCE SCORES")

    cursor.execute(f"""
        SELECT
            object_class,
            COUNT(*) as count,
            AVG(confidence) as avg_conf,
            MIN(confidence) as min_conf,
            MAX(confidence) as max_conf,
            SUM(CASE WHEN confidence = 0.0 THEN 1 ELSE 0 END) as zero_count,
            SUM(CASE WHEN confidence > 0.0 THEN 1 ELSE 0 END) as nonzero_count
        FROM {table_prefix}_detected_objects
        GROUP BY object_class
        ORDER BY count DESC
        LIMIT 20
    """)

    results = cursor.fetchall()
    if results:
        print(f"\n{'Objekt-Klasse':<15} {'Anzahl':>8} {'Ø Conf':>10} {'Min':>8} {'Max':>8} {'Nullen':>8} {'> 0.0':>8}")
        print("-" * 85)

        total_objects = 0
        total_zeros = 0

        for row in results:
            obj = row['object_class']
            count = row['count']
            avg = row['avg_conf'] or 0.0
            min_c = row['min_conf'] or 0.0
            max_c = row['max_conf'] or 0.0
            zeros = row['zero_count'] or 0
            nonzero = row['nonzero_count'] or 0

            total_objects += count
            total_zeros += zeros

            # Highlight wenn alle Nullen sind
            marker = " ⚠️" if zeros == count else ""

            print(f"{obj:<15} {count:>8,} {avg:>10.4f} {min_c:>8.4f} {max_c:>8.4f} {zeros:>8,} {nonzero:>8,}{marker}")

        print(f"\n📊 Zusammenfassung:")
        print(f"   Gesamt Objekte:     {total_objects:,}")
        print(f"   Davon 0.0 Conf:     {total_zeros:,} ({total_zeros/total_objects*100 if total_objects > 0 else 0:.1f}%)")
        print(f"   Davon > 0.0 Conf:   {total_objects-total_zeros:,} ({(total_objects-total_zeros)/total_objects*100 if total_objects > 0 else 0:.1f}%)")

        if total_zeros == total_objects:
            print(f"\n   ❌ PROBLEM: ALLE Confidence Scores sind 0.0!")
        elif total_zeros > 0:
            print(f"\n   ⚠️  WARNUNG: {total_zeros:,} Objekte haben Confidence 0.0")
        else:
            print(f"\n   ✅ GUT: Alle Objekte haben Confidence > 0.0")
    else:
        print("⚠️  Keine Objekte gefunden")

    # Fahrzeug-Statistik
    print_section(f"🚗 {table_prefix.upper()} - FAHRZEUG CONFIDENCE")

    cursor.execute(f"""
        SELECT
            object_class,
            COUNT(*) as count,
            AVG(confidence) as avg_conf,
            COUNT(DISTINCT parking_spot_id) as unique_spots
        FROM {table_prefix}_detected_objects
        WHERE object_class IN ('car', 'truck', 'bus', 'motorcycle', 'bicycle')
        GROUP BY object_class
        ORDER BY count DESC
    """)

    results = cursor.fetchall()
    if results:
        print(f"\n{'Fahrzeug':<15} {'Anzahl':>8} {'Ø Confidence':>14} {'Parkplätze':>12}")
        print("-" * 52)

        for row in results:
            veh = row['object_class']
            count = row['count']
            avg = row['avg_conf'] or 0.0
            spots = row['unique_spots'] or 0

            print(f"{veh:<15} {count:>8,} {avg:>14.4f} {spots:>12}")

    # Recordings-Info
    print_section(f"📹 {table_prefix.upper()} - RECORDINGS INFO")

    cursor.execute(f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN file_type = 'mp4' THEN 1 ELSE 0 END) as videos,
            SUM(CASE WHEN file_type = 'jpg' THEN 1 ELSE 0 END) as images,
            SUM(CASE WHEN analyzed = 1 THEN 1 ELSE 0 END) as analyzed,
            SUM(file_size) as total_size,
            MIN(recorded_at) as first_rec,
            MAX(recorded_at) as last_rec
        FROM {table_prefix}_recordings
    """)

    row = cursor.fetchone()
    if row and row['total'] > 0:
        total = row['total']
        videos = row['videos'] or 0
        images = row['images'] or 0
        analyzed = row['analyzed'] or 0
        size = row['total_size'] or 0

        print(f"\nGesamt Recordings:    {total:,}")
        print(f"  - Videos (MP4):     {videos:,}")
        print(f"  - Bilder (JPG):     {images:,}")
        print(f"  - Analysiert:       {analyzed:,} ({analyzed/total*100:.1f}%)")
        print(f"Gesamtgröße:          {format_size(size)}")
        print(f"Zeitraum:             {row['first_rec']} bis {row['last_rec']}")

    cursor.close()

def compare_schemas(conn):
    """Vergleicht cam_* und cam2_* Schemas"""
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    print_section("🔄 SCHEMA-VERGLEICH: cam_* vs cam2_*")

    # Vergleiche Anzahl Objekte
    cursor.execute("SELECT COUNT(*) as count FROM cam_detected_objects")
    cam_objects = cursor.fetchone()['count']

    cursor.execute("SELECT COUNT(*) as count FROM cam2_detected_objects")
    cam2_objects = cursor.fetchone()['count']

    # Vergleiche Anzahl Recordings
    cursor.execute("SELECT COUNT(*) as count FROM cam_recordings")
    cam_recordings = cursor.fetchone()['count']

    cursor.execute("SELECT COUNT(*) as count FROM cam2_recordings")
    cam2_recordings = cursor.fetchone()['count']

    # Vergleiche Confidence Averages
    cursor.execute("SELECT AVG(confidence) as avg FROM cam_detected_objects")
    cam_avg_conf = cursor.fetchone()['avg'] or 0.0

    cursor.execute("SELECT AVG(confidence) as avg FROM cam2_detected_objects")
    cam2_avg_conf = cursor.fetchone()['avg'] or 0.0

    print(f"\n{'Metrik':<30} {'cam_* (PROD)':>20} {'cam2_* (TEST)':>20}")
    print("-" * 72)
    print(f"{'Recordings':<30} {cam_recordings:>20,} {cam2_recordings:>20,}")
    print(f"{'Detected Objects':<30} {cam_objects:>20,} {cam2_objects:>20,}")
    print(f"{'Ø Confidence':<30} {cam_avg_conf:>20.6f} {cam2_avg_conf:>20.6f}")

    print(f"\n💡 Frontend nutzt:")
    print(f"   - Faces:      cam2_detected_faces")
    print(f"   - Objects:    cam_detected_objects    ← WICHTIG für Confidence!")
    print(f"   - Recordings: cam_recordings          ← WICHTIG für Confidence!")

    cursor.close()

def main():
    print_header("CAM vs CAM2 - CONFIDENCE SCORE REPORT", 80)
    print(f"Zeitpunkt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n🎯 Ziel: Prüfen ob person.py fix die Confidence Scores in PRODUCTION (cam_*) korrigiert hat")

    conn = connect_db()

    try:
        # Check CAM (Production)
        check_confidence_scores(conn, 'cam')

        # Check CAM2 (Test)
        check_confidence_scores(conn, 'cam2')

        # Compare
        compare_schemas(conn)

        print("\n" + "=" * 80)
        print("✓ Report erfolgreich generiert")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()

if __name__ == "__main__":
    main()
