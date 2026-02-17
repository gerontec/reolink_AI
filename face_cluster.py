#!/usr/bin/env python3
"""
Simple Face Clustering Script
Durchsucht alle JPGs, erkennt Gesichter und clustert sie nach Ähnlichkeit
CUDA-Setup wie in person.py
"""

import os
import sys
import glob
import time
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.cluster import DBSCAN

# GPU Configuration (wie in person.py)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# AI/ML imports (gleiche Reihenfolge wie person.py)
import cv2
import torch
import face_recognition

# Pfad zu JPGs
JPG_PATH = '/var/www/web1/2026/02/*.jpg'

def check_gpu():
    """GPU-Status anzeigen (wie in person.py)"""
    print("=" * 60)
    print("GPU STATUS CHECK")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA verfügbar: {cuda_available}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"Anzahl GPUs: {gpu_count}")

        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")

        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("⚠️  CUDA nicht verfügbar - läuft auf CPU")

    print("=" * 60)
    print()
    return cuda_available

def main():
    # GPU Status prüfen
    check_gpu()

    # Start Timestamp
    start_time = time.time()
    print(f"⏱️  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

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
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(jpg_files) - i) / rate if rate > 0 else 0
                print(f"  Verarbeitet: {i}/{len(jpg_files)} | "
                      f"{len(all_encodings)} Gesichter | "
                      f"{rate:.1f} Bilder/s | "
                      f"ETA: {eta:.0f}s")

        except Exception as e:
            print(f"⚠️  Fehler bei {os.path.basename(img_path)}: {e}")

    detection_time = time.time() - start_time
    print(f"\n✅ {len(all_encodings)} Gesichter in {len(jpg_files)} Bildern gefunden")
    print(f"⏱️  Face Detection: {detection_time:.1f}s ({detection_time/len(jpg_files):.2f}s/Bild)\n")

    if len(all_encodings) == 0:
        print("❌ Keine Gesichter gefunden!")
        return

    # DBSCAN Clustering
    print("🧮 Clustere Gesichter...")
    cluster_start = time.time()

    encodings_array = np.array(all_encodings)
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='euclidean').fit(encodings_array)
    labels = clustering.labels_

    cluster_time = time.time() - cluster_start

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)

    print(f"✅ {n_clusters} verschiedene Gesichter identifiziert")
    print(f"   {n_noise} einzelne/unkategorisierte Detektionen")
    print(f"⏱️  Clustering: {cluster_time:.2f}s\n")

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
    total_time = time.time() - start_time

    print("=" * 60)
    print("ZUSAMMENFASSUNG")
    print("=" * 60)
    print(f"Gesamt Bilder:     {len(jpg_files)}")
    print(f"Gesamt Gesichter:  {len(all_encodings)}")
    print(f"Unique Personen:   {n_clusters}")
    print(f"Noise Detektionen: {n_noise}")
    print()
    print("BENCHMARK")
    print("-" * 60)
    print(f"Face Detection:    {detection_time:.1f}s ({detection_time/len(jpg_files):.2f}s/Bild)")
    print(f"Clustering:        {cluster_time:.2f}s")
    print(f"Gesamt-Zeit:       {total_time:.1f}s ({total_time/60:.1f} Min)")
    print(f"Durchschnitt:      {total_time/len(jpg_files):.2f}s/Bild")
    print(f"Ende:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

if __name__ == "__main__":
    main()
