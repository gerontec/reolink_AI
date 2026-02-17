#!/usr/bin/env python3
"""
Schema Migration - Add face clustering columns
Safely adds face_embedding and face_cluster_id to cam2_detected_faces
"""

import pymysql
import sys

DB_CONFIG = {
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

def check_column_exists(cursor, table: str, column: str) -> bool:
    """Prüft ob Spalte existiert"""
    cursor.execute(f"""
        SELECT COUNT(*) as count
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'wagodb'
        AND TABLE_NAME = '{table}'
        AND COLUMN_NAME = '{column}'
    """)
    result = cursor.fetchone()
    return result[0] > 0

def main():
    print("=" * 80)
    print("  CAM2 SCHEMA MIGRATION - Face Clustering Columns")
    print("=" * 80)

    try:
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Check if columns already exist
        has_embedding = check_column_exists(cursor, 'cam2_detected_faces', 'face_embedding')
        has_cluster_id = check_column_exists(cursor, 'cam2_detected_faces', 'face_cluster_id')

        print(f"\nAktueller Status:")
        print(f"  face_embedding:   {'✓ Existiert' if has_embedding else '✗ Fehlt'}")
        print(f"  face_cluster_id:  {'✓ Existiert' if has_cluster_id else '✗ Fehlt'}")

        changes_made = False

        # Add face_embedding if missing
        if not has_embedding:
            print("\n→ Füge face_embedding hinzu...")
            cursor.execute("""
                ALTER TABLE cam2_detected_faces
                ADD COLUMN face_embedding BLOB DEFAULT NULL
                AFTER bbox_y2
            """)
            conn.commit()
            print("  ✓ face_embedding hinzugefügt")
            changes_made = True
        else:
            print("\n→ face_embedding bereits vorhanden (überspringe)")

        # Add face_cluster_id if missing
        if not has_cluster_id:
            print("\n→ Füge face_cluster_id hinzu...")
            cursor.execute("""
                ALTER TABLE cam2_detected_faces
                ADD COLUMN face_cluster_id INT DEFAULT NULL
                AFTER face_embedding
            """)
            conn.commit()
            print("  ✓ face_cluster_id hinzugefügt")
            changes_made = True
        else:
            print("\n→ face_cluster_id bereits vorhanden (überspringe)")

        # Check and add index if missing
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM INFORMATION_SCHEMA.STATISTICS
            WHERE TABLE_SCHEMA = 'wagodb'
            AND TABLE_NAME = 'cam2_detected_faces'
            AND INDEX_NAME = 'idx_cluster'
        """)
        has_index = cursor.fetchone()[0] > 0

        if not has_index and has_cluster_id:
            print("\n→ Erstelle Index idx_cluster...")
            cursor.execute("""
                ALTER TABLE cam2_detected_faces
                ADD INDEX idx_cluster (face_cluster_id)
            """)
            conn.commit()
            print("  ✓ Index idx_cluster erstellt")
            changes_made = True
        else:
            print("\n→ Index idx_cluster bereits vorhanden (überspringe)")

        # Show final schema
        print("\n" + "-" * 80)
        print("  Finales Schema von cam2_detected_faces:")
        print("-" * 80)
        cursor.execute("SHOW COLUMNS FROM cam2_detected_faces")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  {col[0]:20} {col[1]:20} {col[2]:5} {col[3]:5} {col[4] or ''}")

        cursor.close()
        conn.close()

        print("\n" + "=" * 80)
        if changes_made:
            print("  ✓ Migration erfolgreich abgeschlossen!")
            print("=" * 80)
            print("\nNächste Schritte:")
            print("  1. Bilder neu analysieren: python3 person.py --jpg-only --limit 100")
            print("  2. Clustering ausführen:   python3 cam2_cluster_faces.py")
            print("  3. Report ansehen:          python3 cam2_report.py")
        else:
            print("  ✓ Schema bereits aktuell - keine Änderungen nötig")
            print("=" * 80)
        print()

    except Exception as e:
        print(f"\n❌ Fehler bei Migration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
