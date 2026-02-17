#!/usr/bin/env python3
"""
Camera Database Report - Generiert Reports aus allen Kamera-Tabellen
Zeigt Statistiken zu Aufnahmen, erkannten Gesichtern, Objekten und Fahrzeugen
"""

import pymysql
from datetime import datetime, timedelta
from pathlib import Path
import sys
import argparse

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

def get_db_connection():
    """Stellt Datenbankverbindung her"""
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Datenbankverbindung fehlgeschlagen: {e}")
        sys.exit(1)


def format_size(size_bytes):
    if size_bytes is None:
        return "0 B"
    size_bytes = float(size_bytes)
    """Formatiert Bytes zu lesbarer Größe"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def print_section(title):
    """Druckt formatierte Sektion-Überschrift"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def report_recordings_summary(cursor):
    """Zeigt Zusammenfassung aller Aufnahmen"""
    print_section("📹 AUFNAHMEN ÜBERSICHT")
    
    # Gesamt-Statistiken
    query = """
        SELECT 
            COUNT(*) as total_recordings,
            SUM(CASE WHEN file_type = 'jpg' THEN 1 ELSE 0 END) as total_images,
            SUM(CASE WHEN file_type = 'mp4' THEN 1 ELSE 0 END) as total_videos,
            SUM(file_size) as total_size,
            SUM(CASE WHEN analyzed = 1 THEN 1 ELSE 0 END) as analyzed_count,
            MIN(recorded_at) as first_recording,
            MAX(recorded_at) as last_recording
        FROM cam_recordings
    """
    cursor.execute(query)
    result = cursor.fetchone()
    
    if result and result[0] > 0:
        total, images, videos, size, analyzed, first, last = result
        
        print(f"Gesamt Aufnahmen:     {total:,}")
        print(f"  - Bilder (JPG):     {images:,}")
        print(f"  - Videos (MP4):     {videos:,}")
        print(f"Gesamt-Größe:         {format_size(size)}")
        print(f"Analysiert:           {analyzed:,} ({analyzed/total*100:.1f}%)")
        print(f"Zeitraum:             {first} bis {last}")
        
        # Durchschnitte
        if images > 0:
            avg_img_size = size / images if images else 0
            print(f"Ø Bildgröße:          {format_size(avg_img_size)}")
    else:
        print("⚠ Keine Aufnahmen in der Datenbank")
    
    # Pro Kamera
    print("\n📊 Pro Kamera:")
    query = """
        SELECT 
            camera_name,
            COUNT(*) as count,
            SUM(file_size) as size,
            SUM(CASE WHEN file_type = 'jpg' THEN 1 ELSE 0 END) as images,
            SUM(CASE WHEN file_type = 'mp4' THEN 1 ELSE 0 END) as videos
        FROM cam_recordings
        GROUP BY camera_name
        ORDER BY count DESC
    """
    cursor.execute(query)
    
    for row in cursor.fetchall():
        cam, count, size, imgs, vids = row
        print(f"  {cam:12s}: {count:6,} Aufnahmen ({imgs:5,} JPG, {vids:5,} MP4) - {format_size(size)}")


def report_analysis_summary(cursor):
    """Zeigt AI-Analyse Zusammenfassung"""
    print_section("🤖 AI-ANALYSE ÜBERSICHT")
    
    query = """
        SELECT 
            COUNT(*) as total_analyzed,
            SUM(total_faces) as all_faces,
            SUM(total_objects) as all_objects,
            SUM(total_vehicles) as all_vehicles,
            SUM(max_persons) as all_persons,
            SUM(CASE WHEN gpu_used = 1 THEN 1 ELSE 0 END) as gpu_count
        FROM cam_analysis_summary
    """
    cursor.execute(query)
    result = cursor.fetchone()
    
    if result and result[0] > 0:
        total, faces, objects, vehicles, persons, gpu = result
        
        print(f"Analysierte Dateien:  {total:,}")
        print(f"  - Mit GPU:          {gpu:,} ({gpu/total*100:.1f}%)")
        print(f"Erkannte Gesichter:   {faces:,}")
        print(f"Erkannte Objekte:     {objects:,}")
        print(f"Erkannte Fahrzeuge:   {vehicles:,}")
        print(f"Max. Personen:        {persons:,}")
        
        if total > 0:
            print(f"\nØ Pro Analyse:")
            print(f"  - Gesichter:        {faces/total:.2f}")
            print(f"  - Objekte:          {objects/total:.2f}")
            print(f"  - Fahrzeuge:        {vehicles/total:.2f}")
    else:
        print("⚠ Keine analysierten Dateien")


def report_detected_faces(cursor, limit=20):
    """Zeigt erkannte Gesichter"""
    print_section("👤 ERKANNTE PERSONEN (Top {})".format(limit))
    
    # Bekannte Personen
    query = """
        SELECT 
            person_name,
            COUNT(*) as detections,
            AVG(confidence) as avg_confidence,
            MAX(detected_at) as last_seen
        FROM cam_detected_faces
        WHERE person_name != 'Unknown'
        GROUP BY person_name
        ORDER BY detections DESC
        LIMIT %s
    """
    cursor.execute(query, (limit,))
    
    results = cursor.fetchall()
    if results:
        print(f"\n{'Person':<20s} {'Detektionen':>12s} {'Ø Konfidenz':>14s} {'Zuletzt gesehen':<20s}")
        print("-" * 80)
        for row in results:
            name, count, conf, last = row
            print(f"{name:<20s} {count:>12,} {conf:>13.2f} {str(last):<20s}")
    else:
        print("⚠ Keine bekannten Personen erkannt")
    
    # Unbekannte Gesichter
    query = """
        SELECT COUNT(*) 
        FROM cam_detected_faces 
        WHERE person_name = 'Unknown'
    """
    cursor.execute(query)
    unknown_count = cursor.fetchone()[0]
    
    if unknown_count > 0:
        print(f"\n⚠ Unbekannte Gesichter: {unknown_count:,}")


def report_detected_objects(cursor, limit=15):
    """Zeigt erkannte Objekte"""
    print_section("🎯 ERKANNTE OBJEKTE (Top {})".format(limit))
    
    query = """
        SELECT 
            object_class,
            COUNT(*) as detections,
            AVG(confidence) as avg_confidence
        FROM cam_detected_objects
        GROUP BY object_class
        ORDER BY detections DESC
        LIMIT %s
    """
    cursor.execute(query, (limit,))
    
    results = cursor.fetchall()
    if results:
        print(f"\n{'Objekt':<20s} {'Detektionen':>12s} {'Ø Konfidenz':>14s}")
        print("-" * 50)
        for row in results:
            obj_class, count, conf = row
            print(f"{obj_class:<20s} {count:>12,} {conf:>13.2f}")
    else:
        print("⚠ Keine Objekte erkannt")


def report_vehicles(cursor):
    """Zeigt Fahrzeug-Statistiken"""
    print_section("🚗 FAHRZEUGE")
    
    # Fahrzeug-Klassen
    vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
    
    query = """
        SELECT 
            object_class,
            COUNT(*) as detections,
            AVG(confidence) as avg_confidence
        FROM cam_detected_objects
        WHERE object_class IN (%s, %s, %s, %s, %s)
        GROUP BY object_class
        ORDER BY detections DESC
    """
    cursor.execute(query, vehicle_classes)
    
    results = cursor.fetchall()
    if results:
        total_vehicles = sum(row[1] for row in results)
        print(f"\nGesamt Fahrzeug-Detektionen: {total_vehicles:,}\n")
        
        print(f"{'Fahrzeugtyp':<20s} {'Detektionen':>12s} {'Ø Konfidenz':>14s}")
        print("-" * 50)
        for row in results:
            veh_type, count, conf = row
            print(f"{veh_type:<20s} {count:>12,} {conf:>13.2f}")
    else:
        print("⚠ Keine Fahrzeuge erkannt")


def report_recent_activity(cursor, days=7):
    """Zeigt aktivste Tage"""
    print_section(f"📅 AKTIVITÄT (letzte {days} Tage)")
    
    query = """
        SELECT 
            DATE(recorded_at) as date,
            COUNT(*) as recordings,
            SUM(CASE WHEN file_type = 'jpg' THEN 1 ELSE 0 END) as images,
            SUM(CASE WHEN file_type = 'mp4' THEN 1 ELSE 0 END) as videos,
            SUM(file_size) as size
        FROM cam_recordings
        WHERE recorded_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
        GROUP BY DATE(recorded_at)
        ORDER BY date DESC
    """
    cursor.execute(query, (days,))
    
    results = cursor.fetchall()
    if results:
        print(f"\n{'Datum':<12s} {'Aufnahmen':>10s} {'Bilder':>8s} {'Videos':>8s} {'Größe':>12s}")
        print("-" * 60)
        for row in results:
            date, count, imgs, vids, size = row
            print(f"{str(date):<12s} {count:>10,} {imgs:>8,} {vids:>8,} {format_size(size):>12s}")
    else:
        print(f"⚠ Keine Aktivität in den letzten {days} Tagen")


def report_hourly_distribution(cursor):
    """Zeigt Aufnahmen pro Stunde"""
    print_section("🕐 STÜNDLICHE VERTEILUNG")
    
    query = """
        SELECT 
            HOUR(recorded_at) as hour,
            COUNT(*) as recordings
        FROM cam_recordings
        GROUP BY HOUR(recorded_at)
        ORDER BY hour
    """
    cursor.execute(query)
    
    results = cursor.fetchall()
    if results:
        max_count = max(row[1] for row in results)
        
        print()
        for row in results:
            hour, count = row
            bar_length = int((count / max_count) * 50) if max_count > 0 else 0
            bar = "█" * bar_length
            print(f"{hour:02d}:00 {count:>6,} {bar}")
    else:
        print("⚠ Keine Daten verfügbar")


def report_top_cameras_today(cursor):
    """Zeigt aktivste Kameras heute"""
    print_section("📹 AKTIVSTE KAMERAS HEUTE")
    
    query = """
        SELECT 
            camera_name,
            COUNT(*) as recordings,
            SUM(CASE WHEN file_type = 'jpg' THEN 1 ELSE 0 END) as images,
            SUM(CASE WHEN file_type = 'mp4' THEN 1 ELSE 0 END) as videos
        FROM cam_recordings
        WHERE DATE(recorded_at) = CURDATE()
        GROUP BY camera_name
        ORDER BY recordings DESC
    """
    cursor.execute(query)
    
    results = cursor.fetchall()
    if results:
        print(f"\n{'Kamera':<15s} {'Aufnahmen':>10s} {'Bilder':>8s} {'Videos':>8s}")
        print("-" * 45)
        for row in results:
            cam, count, imgs, vids = row
            print(f"{cam:<15s} {count:>10,} {imgs:>8,} {vids:>8,}")
    else:
        print("⚠ Keine Aufnahmen heute")


def report_storage_stats(cursor):
    """Zeigt Speicher-Statistiken"""
    print_section("💾 SPEICHER-STATISTIKEN")
    
    # Gesamt-Größe
    query = "SELECT SUM(file_size) FROM cam_recordings"
    cursor.execute(query)
    total_size = cursor.fetchone()[0] or 0
    
    print(f"\nGesamt-Speicherverbrauch: {format_size(total_size)}")
    
    # Pro Dateityp
    query = """
        SELECT 
            file_type,
            COUNT(*) as count,
            SUM(file_size) as size,
            AVG(file_size) as avg_size
        FROM cam_recordings
        GROUP BY file_type
    """
    cursor.execute(query)
    
    print(f"\n{'Typ':<6s} {'Anzahl':>10s} {'Gesamt':>15s} {'Ø Größe':>15s}")
    print("-" * 50)
    for row in cursor.fetchall():
        ftype, count, size, avg = row
        print(f"{ftype:<6s} {count:>10,} {format_size(size):>15s} {format_size(avg):>15s}")
    
    # Wachstum letzte 7 Tage
    query = """
        SELECT SUM(file_size)
        FROM cam_recordings
        WHERE recorded_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
    """
    cursor.execute(query)
    week_growth = cursor.fetchone()[0] or 0
    
    if week_growth > 0:
        daily_avg = week_growth / 7
        monthly_projection = daily_avg * 30
        print(f"\nWachstum (7 Tage):        {format_size(week_growth)}")
        print(f"Ø pro Tag:                {format_size(daily_avg)}")
        print(f"Projektion (30 Tage):     {format_size(monthly_projection)}")


def report_face_recognition_accuracy(cursor):
    """Zeigt Face Recognition Genauigkeit"""
    print_section("🎯 GESICHTSERKENNUNG GENAUIGKEIT")
    
    # Bekannte vs Unbekannte
    query = """
        SELECT 
            CASE 
                WHEN person_name = 'Unknown' THEN 'Unbekannt'
                ELSE 'Bekannt'
            END as category,
            COUNT(*) as count,
            AVG(confidence) as avg_confidence
        FROM cam_detected_faces
        GROUP BY category
    """
    cursor.execute(query)
    
    results = cursor.fetchall()
    if results:
        print(f"\n{'Kategorie':<15s} {'Anzahl':>10s} {'Ø Konfidenz':>14s}")
        print("-" * 42)
        for row in results:
            cat, count, conf = row
            conf_val = conf if conf is not None else 0.0
            print(f"{cat:<15s} {count:>10,} {conf_val:>13.2f}")
    
    # Konfidenz-Verteilung (nur bekannte Gesichter)
    query = """
        SELECT 
            CASE 
                WHEN confidence >= 0.9 THEN 'Sehr hoch (≥0.9)'
                WHEN confidence >= 0.7 THEN 'Hoch (0.7-0.9)'
                WHEN confidence >= 0.5 THEN 'Mittel (0.5-0.7)'
                ELSE 'Niedrig (<0.5)'
            END as conf_range,
            COUNT(*) as count
        FROM cam_detected_faces
        WHERE person_name != 'Unknown'
        GROUP BY conf_range
        ORDER BY MIN(confidence) DESC
    """
    cursor.execute(query)
    
    results = cursor.fetchall()
    if results:
        print(f"\nKonfidenz-Verteilung (bekannte Personen):")
        print("-" * 42)
        for row in results:
            range_name, count = row
            print(f"  {range_name:<20s} {count:>10,}")


def generate_full_report(output_file=None):
    """Generiert vollständigen Report"""
    
    # Header
    header = f"""
{'=' * 80}
  KAMERA-DATENBANK REPORT
  Generiert: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""
    
    if output_file:
        print(f"📝 Generiere Report: {output_file}")
    
    # Ausgabe-Stream vorbereiten
    if output_file:
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')
    
    print(header)
    
    # Datenbank-Verbindung
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Alle Reports ausführen
        report_recordings_summary(cursor)
        report_analysis_summary(cursor)
        report_detected_faces(cursor)
        report_detected_objects(cursor)
        report_vehicles(cursor)
        report_face_recognition_accuracy(cursor)
        report_recent_activity(cursor, days=7)
        report_top_cameras_today(cursor)
        report_hourly_distribution(cursor)
        report_storage_stats(cursor)
        
        # Footer
        print("\n" + "=" * 80)
        print("  Ende des Reports")
        print("=" * 80)
        
    finally:
        cursor.close()
        conn.close()
        
        if output_file:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"✓ Report gespeichert: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Camera Database Report Generator'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Ausgabedatei (z.B. report.txt). Ohne Angabe wird auf stdout ausgegeben.'
    )
    parser.add_argument(
        '--daily-report',
        action='store_true',
        help='Generiert täglichen Report mit Zeitstempel im Dateinamen'
    )
    parser.add_argument(
        '--report-dir',
        default='./reports',
        help='Verzeichnis für tägliche Reports (default: ./reports)'
    )
    
    args = parser.parse_args()
    
    if args.daily_report:
        # Automatischer Dateiname mit Datum
        report_dir = Path(args.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"cam_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        output_file = report_dir / filename
        
        generate_full_report(str(output_file))
    else:
        generate_full_report(args.output)


if __name__ == "__main__":
    main()
