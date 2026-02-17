#!/usr/bin/env python3
"""
Quick check for confidence scores in cam_* tables
"""
import sys

try:
    import pymysql
    use_pymysql = True
except:
    use_pymysql = False

if not use_pymysql:
    try:
        import mysql.connector
        use_mysql_connector = True
    except:
        print("❌ Keine MySQL-Library verfügbar (pymysql oder mysql.connector)")
        sys.exit(1)
else:
    use_mysql_connector = False

DB_CONFIG = {
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

if use_pymysql:
    conn = pymysql.connect(**DB_CONFIG)
else:
    conn = mysql.connector.connect(**DB_CONFIG)

cursor = conn.cursor()

print("\n" + "="*80)
print("  CAM_* CONFIDENCE SCORE CHECK")
print("="*80)

# Check cam_detected_objects
print("\n📊 cam_detected_objects - Confidence Scores:\n")
cursor.execute("""
    SELECT
        object_class,
        COUNT(*) as count,
        AVG(confidence) as avg_conf,
        MIN(confidence) as min_conf,
        MAX(confidence) as max_conf,
        SUM(CASE WHEN confidence = 0.0 THEN 1 ELSE 0 END) as zero_count
    FROM cam_detected_objects
    GROUP BY object_class
    ORDER BY count DESC
    LIMIT 15
""")

print(f"{'Klasse':<15} {'Anzahl':>8} {'Ø Conf':>10} {'Min':>8} {'Max':>8} {'Nullen':>8}")
print("-"*70)

for row in cursor.fetchall():
    obj, count, avg, min_c, max_c, zeros = row
    print(f"{obj:<15} {count:>8,} {avg:>10.4f} {min_c:>8.4f} {max_c:>8.4f} {zeros:>8,}")

# Check cam_recordings for MP4s
print("\n📹 cam_recordings - MP4 Dateien:\n")
cursor.execute("""
    SELECT
        COUNT(*) as total_mp4,
        SUM(CASE WHEN analyzed = 1 THEN 1 ELSE 0 END) as analyzed,
        MIN(recorded_at) as first_rec,
        MAX(recorded_at) as last_rec
    FROM cam_recordings
    WHERE file_type = 'mp4'
""")

row = cursor.fetchone()
if row:
    total, analyzed, first, last = row
    print(f"Gesamt MP4s:      {total:,}")
    print(f"Analysiert:       {analyzed:,} ({analyzed/total*100 if total > 0 else 0:.1f}%)")
    print(f"Zeitraum:         {first} bis {last}")

print("\n" + "="*80)
print("✓ Check abgeschlossen")
print("="*80 + "\n")

cursor.close()
conn.close()
