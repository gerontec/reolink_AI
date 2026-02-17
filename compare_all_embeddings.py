#!/usr/bin/env python3
"""
Quick script to compare ALL face embeddings and show distances
"""
import pymysql
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

DB_CONFIG = {
    'host': 'localhost',
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

def main():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Alle Embeddings laden + Dateiname via JOIN
    cursor.execute("""
        SELECT
            f.id,
            f.face_embedding,
            f.person_name,
            f.detected_at,
            r.annotated_image_path
        FROM cam2_detected_faces f
        JOIN cam2_recordings r ON f.recording_id = r.id
        WHERE f.face_embedding IS NOT NULL
        ORDER BY f.id
    """)
    rows = cursor.fetchall()

    face_ids = []
    embeddings = []
    metadata = []

    for face_id, embedding_bytes, person_name, detected_at, image_path in rows:
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        face_ids.append(face_id)
        embeddings.append(embedding)
        # Nur Dateiname ohne Pfad
        filename = image_path.split('/')[-1] if image_path else 'N/A'
        metadata.append((person_name, detected_at, filename))

    embeddings_array = np.array(embeddings)

    # Normalisieren
    embeddings_normalized = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)

    # Distanz-Matrix berechnen
    distances = euclidean_distances(embeddings_normalized)

    print("=" * 160)
    print("ALLE EMBEDDING-DISTANZEN (paarweise)")
    print("=" * 160)
    print(f"{'Face 1':<8} {'Name 1':<12} {'Datei 1':<40} {'Face 2':<8} {'Name 2':<12} {'Datei 2':<40} {'Distanz':<10} {'Ähnlich?':<10}")
    print("-" * 160)

    # Alle Paare anzeigen
    pairs = []
    n = len(face_ids)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((
                face_ids[i], metadata[i][0], metadata[i][2],  # id, name, filename
                face_ids[j], metadata[j][0], metadata[j][2],  # id, name, filename
                distances[i, j]
            ))

    # Sortiere nach Distanz
    pairs.sort(key=lambda x: x[6])

    for face1, name1, file1, face2, name2, file2, dist in pairs:
        similar = "✅ < 0.5" if dist <= 0.5 else "❌ > 0.5"
        print(f"{face1:<8} {name1:<12} {file1:<40} {face2:<8} {name2:<12} {file2:<40} {dist:<10.4f} {similar:<10}")

    print("-" * 160)
    close_pairs = sum(1 for p in pairs if p[6] <= 0.5)
    print(f"Paare mit Distanz ≤ 0.5: {close_pairs}/{len(pairs)}")
    print(f"Kleinste Distanz: {pairs[0][6]:.4f} (Face {pairs[0][0]} + {pairs[0][3]})")
    print(f"Größte Distanz: {pairs[-1][6]:.4f} (Face {pairs[-1][0]} + {pairs[-1][3]})")
    print(f"Durchschnitt: {np.mean([p[6] for p in pairs]):.4f}")
    print("=" * 160)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
