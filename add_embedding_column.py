#!/usr/bin/env python3
"""
Fügt die embedding BLOB Spalte zur cam2_detected_faces Tabelle hinzu
"""
import pymysql
import sys

DB_CONFIG = {
    'host': '127.0.0.1',
    'user': 'gh',
    'password': 'Bp123456',
    'database': 'cam2',
    'port': 3306
}

def main():
    try:
        print("Verbinde mit Datenbank...")
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Check if column already exists
        print("Prüfe ob Spalte 'embedding' existiert...")
        cursor.execute("SHOW COLUMNS FROM cam2_detected_faces LIKE 'embedding'")
        result = cursor.fetchone()

        if result:
            print("✓ Spalte 'embedding' existiert bereits")
        else:
            print("Füge Spalte 'embedding' hinzu...")
            cursor.execute("""
                ALTER TABLE cam2_detected_faces
                ADD COLUMN embedding BLOB AFTER bbox_y2
            """)
            conn.commit()
            print("✓ Spalte 'embedding' erfolgreich hinzugefügt!")

        # Show table schema
        print("\n" + "="*60)
        print("Tabellen-Schema cam2_detected_faces:")
        print("="*60)
        cursor.execute("DESCRIBE cam2_detected_faces")

        print(f"{'Feld':<20} {'Typ':<20} {'Null':<5} {'Key':<5} {'Default':<10}")
        print("-"*60)
        for row in cursor.fetchall():
            field, type_, null, key, default, extra = row
            default_str = str(default) if default else ''
            print(f"{field:<20} {type_:<20} {null:<5} {key:<5} {default_str:<10}")

        cursor.close()
        conn.close()

        print("\n✓ Erfolgreich abgeschlossen!")
        return 0

    except Exception as e:
        print(f"\n✗ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
