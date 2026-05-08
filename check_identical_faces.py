#!/usr/bin/env python3
"""Quick check for the two identical Schorsch faces"""
import pymysql
import numpy as np

DB_CONFIG = {
    'host': 'localhost',
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

conn = pymysql.connect(**DB_CONFIG)
cursor = conn.cursor()

cursor.execute("""
    SELECT
        f.id,
        f.person_name,
        f.confidence,
        f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
        f.detected_at,
        r.recorded_at,
        r.annotated_image_path,
        r.video_file_path,
        LENGTH(f.face_embedding) as embedding_bytes,
        f.face_embedding
    FROM cam2_detected_faces f
    JOIN cam2_recordings r ON f.recording_id = r.id
    WHERE f.id IN (2460, 2461)
    ORDER BY f.id
""")

rows = cursor.fetchall()

print("=" * 120)
print("VERGLEICH DER IDENTISCHEN SCHORSCH-FACES (Distanz 0.0)")
print("=" * 120)

faces = []
for row in rows:
    face_id, name, conf, x1, y1, x2, y2, detected_at, recorded_at, img_path, vid_path, emb_size, emb_bytes = row
    embedding = np.frombuffer(emb_bytes, dtype=np.float32)

    faces.append({
        'id': face_id,
        'name': name,
        'confidence': conf,
        'bbox': (x1, y1, x2, y2),
        'detected_at': detected_at,
        'recorded_at': recorded_at,
        'image': img_path,
        'video': vid_path,
        'embedding': embedding,
        'emb_size': emb_size
    })

    print(f"\nFace ID {face_id}:")
    print(f"  Name: {name}")
    print(f"  Confidence: {conf}")
    print(f"  BBox: ({x1}, {y1}, {x2}, {y2}) - Größe: {x2-x1}x{y2-y1}")
    print(f"  Detected: {detected_at}")
    print(f"  Recorded: {recorded_at}")
    print(f"  Image: {img_path}")
    print(f"  Video: {vid_path}")
    print(f"  Embedding: {emb_size} bytes = {emb_size//4} floats")

# Vergleiche die Embeddings
if len(faces) == 2:
    emb1 = faces[0]['embedding']
    emb2 = faces[1]['embedding']

    print("\n" + "-" * 120)
    print("EMBEDDING-VERGLEICH:")
    print("-" * 120)

    # Sind sie wirklich EXAKT identisch?
    if np.array_equal(emb1, emb2):
        print("⚠️  Embeddings sind BYTE-FÜR-BYTE IDENTISCH!")
        print("    → Das ist sehr ungewöhnlich für 2 verschiedene Frames")
        print("    → Möglicher Bug: Gleiches Embedding 2x gespeichert?")
    else:
        diff = np.abs(emb1 - emb2)
        print(f"Unterschiede gefunden:")
        print(f"  Max Differenz: {np.max(diff):.8f}")
        print(f"  Durchschnitt: {np.mean(diff):.8f}")
        print(f"  Anzahl unterschiedlicher Werte: {np.sum(diff > 0)}")

    # BBox-Vergleich
    bbox1 = faces[0]['bbox']
    bbox2 = faces[1]['bbox']
    if bbox1 == bbox2:
        print(f"\n⚠️  BBoxes sind IDENTISCH: {bbox1}")
    else:
        print(f"\nBBox unterschiedlich:")
        print(f"  Face 2460: {bbox1}")
        print(f"  Face 2461: {bbox2}")

print("=" * 120)

cursor.close()
conn.close()
