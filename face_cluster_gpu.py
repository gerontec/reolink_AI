#!/usr/bin/env python3
"""
GPU-Accelerated Face Clustering with InsightFace
CUDA-optimiert für Tesla P4 - 10-50x schneller als dlib
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

# AI/ML imports
import cv2
import torch
from insightface.app import FaceAnalysis

# Pfad zu JPGs
JPG_PATH = '/var/www/web1/2026/02/*.jpg'

def check_gpu():
    """GPU-Status anzeigen"""
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

def init_insightface(use_gpu=True):
    """
    Initialisiert InsightFace mit GPU-Support

    Models:
    - buffalo_l: Beste Accuracy, langsamer
    - buffalo_s: Guter Kompromiss (empfohlen)
    - buffalo_sc: Schnellste, weniger genau
    """
    print("🔧 Initialisiere InsightFace (GPU)...")

    # FaceAnalysis mit GPU
    app = FaceAnalysis(
        name='buffalo_s',  # Guter Kompromiss zwischen Speed und Accuracy
        providers=['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    )

    # ctx_id = 0 bedeutet GPU 0
    app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))

    print(f"✅ InsightFace bereit - Device: {'GPU' if use_gpu else 'CPU'}")
    print(f"   Model: buffalo_s")
    print(f"   Detection Size: 640x640\n")

    return app

def main():
    # GPU Status prüfen
    gpu_available = check_gpu()

    # Start Timestamp
    start_time = time.time()
    print(f"⏱️  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # InsightFace initialisieren
    try:
        face_app = init_insightface(use_gpu=gpu_available)
    except Exception as e:
        print(f"❌ Fehler beim Laden von InsightFace: {e}")
        print("   Versuche CPU-Modus...")
        face_app = init_insightface(use_gpu=False)

    # Alle JPGs finden
    jpg_files = sorted(glob.glob(JPG_PATH))
    print(f"🔍 Analysiere {len(jpg_files)} Bilder...\n")

    all_embeddings = []
    all_files = []
    face_count = 0

    # Face Detection + Encoding mit GPU
    for i, img_path in enumerate(jpg_files, 1):
        try:
            # Bild laden mit OpenCV
            image = cv2.imread(img_path)

            if image is None:
                print(f"⚠️  Konnte nicht laden: {os.path.basename(img_path)}")
                continue

            # Face Detection + Embedding (GPU-beschleunigt!)
            faces = face_app.get(image)

            # Embeddings extrahieren
            for face in faces:
                # face.embedding ist 512-dim ArcFace embedding
                all_embeddings.append(face.embedding)
                all_files.append(img_path)
                face_count += 1

            # Progress Update
            if i % 20 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(jpg_files) - i) / rate if rate > 0 else 0
                print(f"  Verarbeitet: {i}/{len(jpg_files)} | "
                      f"{face_count} Gesichter | "
                      f"{rate:.1f} Bilder/s | "
                      f"ETA: {eta:.0f}s")

        except Exception as e:
            print(f"⚠️  Fehler bei {os.path.basename(img_path)}: {e}")

    detection_time = time.time() - start_time
    print(f"\n✅ {face_count} Gesichter in {len(jpg_files)} Bildern gefunden")
    print(f"⏱️  Face Detection: {detection_time:.1f}s ({detection_time/len(jpg_files):.3f}s/Bild)\n")

    if len(all_embeddings) == 0:
        print("❌ Keine Gesichter gefunden!")
        return

    # DBSCAN Clustering
    print("🧮 Clustere Gesichter...")
    cluster_start = time.time()

    embeddings_array = np.array(all_embeddings)

    # InsightFace embeddings sind normalized, verwende cosine distance
    # eps=0.4-0.6 für cosine distance bei normalisierten Embeddings
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(embeddings_array)
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
    print("FACE CLUSTERING ERGEBNISSE (GPU)")
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
    print(f"Gesamt Gesichter:  {face_count}")
    print(f"Unique Personen:   {n_clusters}")
    print(f"Noise Detektionen: {n_noise}")
    print()
    print("BENCHMARK (GPU)")
    print("-" * 60)
    print(f"Face Detection:    {detection_time:.1f}s ({detection_time/len(jpg_files):.3f}s/Bild)")
    print(f"Clustering:        {cluster_time:.2f}s")
    print(f"Gesamt-Zeit:       {total_time:.1f}s ({total_time/60:.1f} Min)")
    print(f"Durchschnitt:      {total_time/len(jpg_files):.3f}s/Bild")
    print(f"Speedup:           ~{150/total_time:.0f}x schneller als CPU (geschätzt)")
    print(f"Ende:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

if __name__ == "__main__":
    main()
