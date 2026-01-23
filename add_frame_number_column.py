#!/usr/bin/env python3
"""
Fügt die frame_number Spalte zur cam2_detected_faces Tabelle hinzu
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
        print("Prüfe ob Spalte 'frame_number' existiert...")
        cursor.execute("SHOW COLUMNS FROM cam2_detected_faces LIKE 'frame_number'")
        result = cursor.fetchone()

        if result:
            print("✓ Spalte 'frame_number' existiert bereits")
        else:
            print("Füge Spalte 'frame_number' hinzu...")
            cursor.execute("""
                ALTER TABLE cam2_detected_faces
                ADD COLUMN frame_number INT DEFAULT 0 AFTER embedding
            """)
            conn.commit()
            print("✓ Spalte 'frame_number' erfolgreich hinzugefügt!")

        # Show table schema
        print("\n" + "="*70)
        print("Tabellen-Schema cam2_detected_faces:")
        print("="*70)
        cursor.execute("DESCRIBE cam2_detected_faces")

        print(f"{'Feld':<20} {'Typ':<25} {'Null':<5} {'Key':<5} {'Default':<10}")
        print("-"*70)
        for row in cursor.fetchall():
            field, type_, null, key, default, extra = row
            default_str = str(default) if default else ''
            print(f"{field:<20} {type_:<25} {null:<5} {key:<5} {default_str:<10}")

        cursor.close()
        conn.close()

        print("\n✓ Erfolgreich abgeschlossen!")
        print("\nHinweis: Bestehende Gesichter haben frame_number = 0 (erster Frame)")
        print("Neue Analysen speichern den korrekten Frame für beste Qualität.")
        return 0

    except Exception as e:
        print(f"\n✗ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
