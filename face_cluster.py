#!/usr/bin/env python3
"""
Simple Face Clustering Script
Durchsucht alle JPGs, erkennt Gesichter und clustert sie nach Ähnlichkeit
"""

import face_recognition
import glob
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN
import os

# Pfad zu JPGs
JPG_PATH = '/var/www/web1/2026/02/*.jpg'

def main():
    # Alle JPGs finden
    jpg_files = sorted(glob.glob(JPG_PATH))
    print(f"🔍 Analysiere {len(jpg_files)} Bilder...\n")

    all_encodings = []
    all_files = []

    # Face Detection
    for i, img_path in enumerate(jpg_files, 1):
        try:
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            for encoding in encodings:
                all_encodings.append(encoding)
                all_files.append(img_path)

            if i % 20 == 0:
                print(f"  Verarbeitet: {i}/{len(jpg_files)} Bilder, {len(all_encodings)} Gesichter gefunden")

        except Exception as e:
            print(f"⚠️  Fehler bei {os.path.basename(img_path)}: {e}")

    print(f"\n✅ {len(all_encodings)} Gesichter in {len(jpg_files)} Bildern gefunden\n")

    if len(all_encodings) == 0:
        print("❌ Keine Gesichter gefunden!")
        return

    # DBSCAN Clustering
    print("🧮 Clustere Gesichter...")
    encodings_array = np.array(all_encodings)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(encodings_array)
    labels = clustering.labels_

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)

    print(f"✅ {n_clusters} verschiedene Gesichter identifiziert")
    print(f"   {n_noise} einzelne/unkategorisierte Detektionen\n")

    # Gruppiere nach Cluster-ID
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append(all_files[i])

    # Ausgabe sortiert nach Häufigkeit
    print("=" * 60)
    print("FACE CLUSTERING ERGEBNISSE")
    print("=" * 60)
    print()

    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)

    for label, files in sorted_clusters:
        if label == -1:
            cluster_name = "NOISE/EINZELN"
        else:
            cluster_name = f"Face ID {label}"

        print(f"📊 {cluster_name}: {len(files)} Detektionen")
        print("-" * 60)

        # Zeige erste 5 Beispiele
        for filepath in files[:5]:
            filename = os.path.basename(filepath)
            print(f"   • {filename}")

        if len(files) > 5:
            print(f"   ... und {len(files)-5} weitere Bilder")

        print()

    # Zusammenfassung
    print("=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"Gesamt Bilder:     {len(jpg_files)}")
    print(f"Gesamt Gesichter:  {len(all_encodings)}")
    print(f"Unique Personen:   {n_clusters}")
    print(f"Noise Detektionen: {n_noise}")
    print()

if __name__ == "__main__":
    main()
