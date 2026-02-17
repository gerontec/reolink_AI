#!/usr/bin/env python3
"""
CAM2 Face Clustering - Groups identical unknown faces using DBSCAN
Analyzes face embeddings from cam2_detected_faces and assigns cluster IDs
"""

import pymysql
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from typing import List, Tuple
import sys
import logging

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    'host': 'localhost',
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

def connect_db():
    """Verbindet mit der Datenbank"""
    try:
        return pymysql.connect(**DB_CONFIG)
    except Exception as e:
        logger.error(f"❌ Datenbankverbindung fehlgeschlagen: {e}")
        sys.exit(1)

def load_face_embeddings(conn) -> Tuple[List[int], np.ndarray]:
    """Lädt alle Face Embeddings aus der Datenbank"""
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, face_embedding
        FROM cam2_detected_faces
        WHERE face_embedding IS NOT NULL
        ORDER BY id
    """)

    rows = cursor.fetchall()
    cursor.close()

    if not rows:
        logger.warning("Keine Face Embeddings gefunden!")
        return [], np.array([])

    face_ids = []
    embeddings = []

    for face_id, embedding_bytes in rows:
        # BLOB zurück zu numpy array konvertieren (512-dim float32)
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

        # Sanity check: ArcFace embeddings sind 512-dimensional
        if len(embedding) != 512:
            logger.warning(f"Face ID {face_id}: Ungültige Embedding-Größe {len(embedding)}, erwartet 512")
            continue

        face_ids.append(face_id)
        embeddings.append(embedding)

    embeddings_array = np.array(embeddings)
    logger.info(f"✓ {len(face_ids)} Face Embeddings geladen (512-dim ArcFace)")

    return face_ids, embeddings_array

def cluster_faces(embeddings: np.ndarray, eps: float = 0.5, min_samples: int = 2) -> np.ndarray:
    """
    Clustert Gesichter mit DBSCAN basierend auf Cosine Distance

    Args:
        embeddings: numpy array (N, 512) mit Face Embeddings
        eps: Maximum distance für Cluster (ArcFace: 0.4-0.6 ist gut)
        min_samples: Mindestanzahl Samples pro Cluster

    Returns:
        cluster_labels: numpy array mit Cluster-IDs (-1 = Noise/Outlier)
    """

    if len(embeddings) == 0:
        return np.array([])

    logger.info(f"Starte DBSCAN Clustering (eps={eps}, min_samples={min_samples})...")

    # DBSCAN mit Cosine Distance
    # ArcFace Embeddings sind normalized, daher: cosine_distance = 1 - dot_product
    # Aber DBSCAN erwartet Euclidean Distance, also konvertieren wir:
    # Für normalized vectors: euclidean_distance² = 2 * (1 - cosine_similarity)
    # Deshalb normalisieren wir erst die Embeddings (falls nicht schon normalized)

    # Embeddings normalisieren (L2-Norm)
    embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # DBSCAN mit Euclidean Distance (auf normalized embeddings = approximiert Cosine)
    # eps Parameter: 0.5 entspricht ca. cosine similarity von 0.875 (sehr ähnlich)
    # Kleinere eps = strengere Cluster
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric='euclidean',  # Auf normalized vectors ≈ Cosine Distance
        n_jobs=-1  # Alle CPU-Kerne nutzen
    )

    cluster_labels = dbscan.fit_predict(embeddings_normalized)

    # Statistiken
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    logger.info(f"✓ Clustering abgeschlossen:")
    logger.info(f"  Cluster gefunden: {n_clusters}")
    logger.info(f"  Noise/Outlier: {n_noise}")

    # Silhouette Score (nur wenn genug Cluster vorhanden)
    if n_clusters > 1 and n_noise < len(cluster_labels):
        try:
            score = silhouette_score(embeddings_normalized, cluster_labels)
            logger.info(f"  Silhouette Score: {score:.3f} (0=schlecht, 1=perfekt)")
        except:
            pass  # Silhouette Score kann manchmal fehlschlagen

    return cluster_labels

def update_cluster_ids(conn, face_ids: List[int], cluster_labels: np.ndarray, reset_first: bool = True):
    """Updated face_cluster_id in der Datenbank"""

    if len(face_ids) != len(cluster_labels):
        logger.error("Fehler: face_ids und cluster_labels haben unterschiedliche Längen!")
        return

    cursor = conn.cursor()

    # Optional: Alle cluster_ids erst auf NULL setzen
    if reset_first:
        logger.info("Setze alle face_cluster_id auf NULL...")
        cursor.execute("UPDATE cam2_detected_faces SET face_cluster_id = NULL")
        conn.commit()

    # Cluster IDs updaten
    # -1 (Noise) wird als NULL gespeichert
    logger.info("Schreibe Cluster-IDs in Datenbank...")

    updated_count = 0
    for face_id, cluster_id in zip(face_ids, cluster_labels):
        if cluster_id == -1:
            # Noise/Outlier → cluster_id = NULL
            cursor.execute(
                "UPDATE cam2_detected_faces SET face_cluster_id = NULL WHERE id = %s",
                (face_id,)
            )
        else:
            # Cluster ID speichern (offset +1 damit Cluster bei 1 starten)
            cursor.execute(
                "UPDATE cam2_detected_faces SET face_cluster_id = %s WHERE id = %s",
                (int(cluster_id) + 1, face_id)
            )
        updated_count += 1

        if updated_count % 500 == 0:
            logger.info(f"  {updated_count}/{len(face_ids)} aktualisiert...")
            conn.commit()

    conn.commit()
    cursor.close()

    logger.info(f"✓ {updated_count} Face Cluster-IDs gespeichert")

def print_cluster_statistics(conn):
    """Zeigt Cluster-Statistiken"""
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # Gesamt-Statistik
    cursor.execute("""
        SELECT
            COUNT(*) as total_faces,
            COUNT(DISTINCT face_cluster_id) as total_clusters,
            SUM(CASE WHEN face_cluster_id IS NULL THEN 1 ELSE 0 END) as unclustered,
            SUM(CASE WHEN face_cluster_id IS NOT NULL THEN 1 ELSE 0 END) as clustered
        FROM cam2_detected_faces
        WHERE face_embedding IS NOT NULL
    """)
    stats = cursor.fetchone()

    print("\n" + "=" * 80)
    print("  CLUSTER STATISTIKEN")
    print("=" * 80)
    print(f"Gesamt Gesichter:     {stats['total_faces']:,}")
    print(f"Cluster gefunden:     {stats['total_clusters']:,}")
    print(f"Geclustert:           {stats['clustered']:,}")
    print(f"Einzelgänger (Noise): {stats['unclustered']:,}")

    # Top 20 Cluster nach Größe
    cursor.execute("""
        SELECT
            face_cluster_id,
            COUNT(*) as size,
            MIN(detected_at) as first_seen,
            MAX(detected_at) as last_seen
        FROM cam2_detected_faces
        WHERE face_cluster_id IS NOT NULL
        GROUP BY face_cluster_id
        ORDER BY size DESC
        LIMIT 20
    """)
    top_clusters = cursor.fetchall()

    if top_clusters:
        print("\n" + "-" * 80)
        print("  TOP 20 CLUSTER (nach Größe)")
        print("-" * 80)
        print(f"{'Cluster':<10} {'Größe':<10} {'Erste Detection':<20} {'Letzte Detection':<20}")
        print("-" * 80)

        for cluster in top_clusters:
            print(f"#{cluster['face_cluster_id']:<9} "
                  f"{cluster['size']:<10} "
                  f"{str(cluster['first_seen']):<20} "
                  f"{str(cluster['last_seen']):<20}")

    # Cluster-Größen Verteilung
    cursor.execute("""
        SELECT
            CASE
                WHEN size = 1 THEN '1 Gesicht'
                WHEN size BETWEEN 2 AND 5 THEN '2-5 Gesichter'
                WHEN size BETWEEN 6 AND 10 THEN '6-10 Gesichter'
                WHEN size BETWEEN 11 AND 50 THEN '11-50 Gesichter'
                WHEN size BETWEEN 51 AND 100 THEN '51-100 Gesichter'
                ELSE '100+ Gesichter'
            END as size_range,
            COUNT(*) as cluster_count,
            SUM(size) as total_faces
        FROM (
            SELECT face_cluster_id, COUNT(*) as size
            FROM cam2_detected_faces
            WHERE face_cluster_id IS NOT NULL
            GROUP BY face_cluster_id
        ) as cluster_sizes
        GROUP BY size_range
        ORDER BY MIN(size)
    """)
    distribution = cursor.fetchall()

    if distribution:
        print("\n" + "-" * 80)
        print("  CLUSTER-GRÖSSEN VERTEILUNG")
        print("-" * 80)
        print(f"{'Größe':<20} {'Anzahl Cluster':<20} {'Gesamt Gesichter':<20}")
        print("-" * 80)

        for row in distribution:
            print(f"{row['size_range']:<20} {row['cluster_count']:<20} {row['total_faces']:<20}")

    print("=" * 80)

    cursor.close()

def main():
    """Main Clustering Workflow"""

    logger.info("=" * 80)
    logger.info("CAM2 Face Clustering - DBSCAN Similarity Clustering")
    logger.info("=" * 80)

    # Parameter
    EPS = 0.5  # Cosine Distance Threshold (kleiner = strengere Cluster)
    MIN_SAMPLES = 2  # Mindestens 2 Gesichter pro Cluster

    logger.info(f"Parameter: eps={EPS}, min_samples={MIN_SAMPLES}")

    # Datenbank verbinden
    conn = connect_db()

    try:
        # 1. Embeddings laden
        face_ids, embeddings = load_face_embeddings(conn)

        if len(face_ids) == 0:
            logger.error("Keine Embeddings vorhanden - abgebrochen")
            return

        # 2. Clustering durchführen
        cluster_labels = cluster_faces(embeddings, eps=EPS, min_samples=MIN_SAMPLES)

        # 3. Cluster-IDs in DB schreiben
        update_cluster_ids(conn, face_ids, cluster_labels, reset_first=True)

        # 4. Statistiken anzeigen
        print_cluster_statistics(conn)

        logger.info("✓ Clustering erfolgreich abgeschlossen")

    except Exception as e:
        logger.error(f"❌ Fehler beim Clustering: {e}")
        import traceback
        traceback.print_exc()

    finally:
        conn.close()

if __name__ == "__main__":
    main()
