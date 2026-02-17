#!/usr/bin/env python3
"""Quick check: Are embeddings being saved?"""

import pymysql

DB_CONFIG = {
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

conn = pymysql.connect(**DB_CONFIG)
cursor = conn.cursor(pymysql.cursors.DictCursor)

# Check total faces
cursor.execute("SELECT COUNT(*) as total FROM cam2_detected_faces")
total = cursor.fetchone()['total']

# Check faces with embeddings
cursor.execute("SELECT COUNT(*) as with_emb FROM cam2_detected_faces WHERE face_embedding IS NOT NULL")
with_emb = cursor.fetchone()['with_emb']

# Check recent faces (last hour)
cursor.execute("""
    SELECT COUNT(*) as recent_total,
           SUM(CASE WHEN face_embedding IS NOT NULL THEN 1 ELSE 0 END) as recent_with_emb
    FROM cam2_detected_faces
    WHERE detected_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
""")
recent = cursor.fetchone()

# Show sample with/without embedding
cursor.execute("""
    SELECT id, person_name, detected_at,
           CASE WHEN face_embedding IS NOT NULL THEN 'YES' ELSE 'NO' END as has_embedding
    FROM cam2_detected_faces
    ORDER BY detected_at DESC
    LIMIT 10
""")
samples = cursor.fetchall()

print("=" * 80)
print("  EMBEDDING CHECK")
print("=" * 80)
print(f"Gesamt Gesichter:           {total:,}")
print(f"Mit Embedding:              {with_emb:,} ({with_emb/total*100:.1f}%)")
print(f"Ohne Embedding:             {total - with_emb:,}")
print()
print(f"Letzte Stunde:")
print(f"  Neue Gesichter:           {recent['recent_total']:,}")
print(f"  Davon mit Embedding:      {recent['recent_with_emb'] or 0:,}")
print()
print("Letzte 10 Detections:")
print("-" * 80)
for s in samples:
    emb_status = "✓ EMBEDDING" if s['has_embedding'] == 'YES' else "✗ KEIN EMB"
    print(f"  #{s['id']:<6} {s['detected_at']}  {emb_status:15} {s['person_name']}")
print("=" * 80)

cursor.close()
conn.close()
