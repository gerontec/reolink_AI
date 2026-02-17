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

    # Alle Embeddings laden
    cursor.execute("""
        SELECT id, face_embedding, person_name, detected_at
        FROM cam2_detected_faces
        WHERE face_embedding IS NOT NULL
        ORDER BY id
    """)
    rows = cursor.fetchall()

    face_ids = []
    embeddings = []
    metadata = []

    for face_id, embedding_bytes, person_name, detected_at in rows:
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        face_ids.append(face_id)
        embeddings.append(embedding)
        metadata.append((person_name, detected_at))

    embeddings_array = np.array(embeddings)

    # Normalisieren
    embeddings_normalized = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)

    # Distanz-Matrix berechnen
    distances = euclidean_distances(embeddings_normalized)

    print("=" * 120)
    print("ALLE EMBEDDING-DISTANZEN (paarweise)")
    print("=" * 120)
    print(f"{'Face 1':<8} {'Name 1':<15} {'Face 2':<8} {'Name 2':<15} {'Distanz':<10} {'Ähnlich?':<15}")
    print("-" * 120)

    # Alle Paare anzeigen
    pairs = []
    n = len(face_ids)
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((
                face_ids[i], metadata[i][0],
                face_ids[j], metadata[j][0],
                distances[i, j]
            ))

    # Sortiere nach Distanz
    pairs.sort(key=lambda x: x[4])

    for face1, name1, face2, name2, dist in pairs:
        similar = "✅ Sehr ähnlich" if dist <= 0.5 else "❌ Verschieden"
        print(f"{face1:<8} {name1:<15} {face2:<8} {name2:<15} {dist:<10.4f} {similar:<15}")

    print("-" * 120)
    close_pairs = sum(1 for p in pairs if p[4] <= 0.5)
    print(f"Paare mit Distanz ≤ 0.5: {close_pairs}/{len(pairs)}")
    print(f"Kleinste Distanz: {pairs[0][4]:.4f}")
    print(f"Größte Distanz: {pairs[-1][4]:.4f}")
    print(f"Durchschnitt: {np.mean([p[4] for p in pairs]):.4f}")
    print("=" * 120)

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
