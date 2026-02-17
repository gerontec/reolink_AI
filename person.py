#!/usr/bin/env python3
"""
AI Video Processor - GPU-Optimiert für Tesla P4
Verarbeitet vorhandene Video/Bild-Dateien mit vollständiger AI-Analyse
Gesichtserkennung, Objekt-Detektion, Szenen-Erkennung
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import re
import json

# Database imports
import mysql.connector
from mysql.connector import Error as MySQLError

# AI/ML imports
import cv2
import numpy as np
from ultralytics import YOLO
import face_recognition
import torch

# Configuration
MEDIA_BASE_PATH = "/var/www/web1"
DB_CONFIG = {
    'host': 'localhost',
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

# AI Model paths
YOLO_MODEL_PATH = "/opt/models/yolov8m.pt"
KNOWN_FACES_DIR = "/opt/known_faces"

# GPU Configuration
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Explizit GPU 0 verwenden

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/gh/python/logs/reolink_processor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Überprüft GPU-Verfügbarkeit und zeigt Details"""
    logger.info("=" * 70)
    logger.info("GPU STATUS CHECK")
    logger.info("=" * 70)
    
    # CUDA verfügbar?
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA verfügbar: {cuda_available}")
    
    if cuda_available:
        # GPU Details
        gpu_count = torch.cuda.device_count()
        logger.info(f"Anzahl GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        
        # Aktuelle GPU
        current_device = torch.cuda.current_device()
        logger.info(f"Aktuelle GPU: {current_device}")
        
        # Memory Info
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        logger.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        
        # CUDA Version
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
    else:
        logger.warning("⚠ CUDA nicht verfügbar! Läuft auf CPU.")
        logger.warning("Bitte PyTorch mit CUDA-Support installieren:")
        logger.warning("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    logger.info("=" * 70)
    
    return cuda_available


class AIAnalyzer:
    """AI-Analyse Engine für Video und Bilder - GPU-Optimiert"""
    
    def __init__(self, yolo_model_path: str, known_faces_dir: str, force_gpu: bool = True):
        self.yolo_model = None
        self.known_faces_dir = Path(known_faces_dir)
        self.known_face_encodings = []
        self.known_face_names = []
        self.force_gpu = force_gpu
        
        # GPU-Device bestimmen
        if torch.cuda.is_available() and force_gpu:
            self.device = 'cuda'
            self.gpu_id = 0
            # GPU für diese Session reservieren
            torch.cuda.set_device(self.gpu_id)
        else:
            self.device = 'cpu'
            self.gpu_id = None
            if force_gpu:
                logger.error("GPU erzwungen, aber CUDA nicht verfügbar!")
                sys.exit(1)
        
        logger.info(f"AI-Analyzer initialisiert - Device: {self.device}")
        
        # YOLO Model laden mit expliziter GPU-Zuweisung
        self._load_yolo_model(yolo_model_path)
        
        # Bekannte Gesichter laden
        self._load_known_faces()
        
        # GPU Warmup
        if self.device == 'cuda':
            self._gpu_warmup()
    
    def _load_yolo_model(self, model_path: str):
        """Lädt YOLO Model explizit auf GPU"""
        try:
            logger.info(f"Lade YOLO Model: {model_path}")
            self.yolo_model = YOLO(model_path)
            
            # Explizit auf GPU verschieben
            if self.device == 'cuda':
                self.yolo_model.to(self.device)
                logger.info(f"✓ YOLO Model auf GPU {self.gpu_id} geladen")
                
                # Model-Infos
                logger.info(f"Model-Typ: {type(self.yolo_model.model)}")
                logger.info(f"Model-Device: {next(self.yolo_model.model.parameters()).device}")
            else:
                logger.info("YOLO Model auf CPU geladen")
                
        except Exception as e:
            logger.error(f"YOLO Model konnte nicht geladen werden: {e}")
            raise
    
    def _gpu_warmup(self):
        """Warmup für GPU - Erste Inferenz ist oft langsam"""
        logger.info("GPU Warmup...")
        try:
            # Dummy-Bild für Warmup
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # YOLO Warmup
            _ = self.yolo_model(dummy_image, verbose=False, device=self.device)
            
            logger.info("✓ GPU Warmup abgeschlossen")
            
            # Memory nach Warmup
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**2
                logger.info(f"GPU Memory nach Warmup: {allocated:.2f} MB")
                
        except Exception as e:
            logger.error(f"GPU Warmup fehlgeschlagen: {e}")
    
    def _load_known_faces(self):
        """Lädt alle bekannten Gesichter aus dem Verzeichnis"""
        if not self.known_faces_dir.exists():
            logger.warning(f"Verzeichnis für bekannte Gesichter nicht gefunden: {self.known_faces_dir}")
            return
        
        face_files = list(self.known_faces_dir.glob("*.jpg")) + list(self.known_faces_dir.glob("*.png"))
        
        if not face_files:
            logger.warning(f"Keine Gesichts-Bilder in {self.known_faces_dir} gefunden")
            return
        
        logger.info(f"Lade {len(face_files)} bekannte Gesichter...")
        
        for image_path in face_files:
            try:
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image)
                
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    # Name ist Dateiname ohne Extension
                    name = image_path.stem.replace('_', ' ')
                    self.known_face_names.append(name)
                    logger.info(f"  ✓ {name}")
                else:
                    logger.warning(f"  ⚠ Kein Gesicht in: {image_path.name}")
            except Exception as e:
                logger.error(f"  ✗ Fehler bei {image_path.name}: {e}")
        
        logger.info(f"✓ {len(self.known_face_names)} bekannte Gesichter geladen")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Gibt aktuelle GPU-Statistiken zurück"""
        if not torch.cuda.is_available():
            return {'available': False}
        
        return {
            'available': True,
            'device': self.gpu_id,
            'name': torch.cuda.get_device_name(self.gpu_id),
            'memory_allocated_mb': torch.cuda.memory_allocated(self.gpu_id) / 1024**2,
            'memory_reserved_mb': torch.cuda.memory_reserved(self.gpu_id) / 1024**2,
            'memory_total_gb': torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3
        }
    
    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Analysiert ein einzelnes Bild"""
        results = {
            'faces': [],
            'objects': [],
            'vehicles': [],
            'persons': 0,
            'analysis_timestamp': datetime.now(),
            'gpu_used': self.device == 'cuda'
        }
        
        try:
            # Bild laden
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Bild konnte nicht geladen werden: {image_path}")
                return results
            
            # Gesichtserkennung (läuft auf CPU - face_recognition unterstützt kein CUDA)
            face_results = self._detect_faces(image)
            results['faces'] = face_results
            
            # Objekt-Detektion mit YOLO (läuft auf GPU)
            yolo_results = self._detect_objects(image)
            results['objects'] = yolo_results['objects']
            results['vehicles'] = yolo_results['vehicles']
            results['persons'] = yolo_results['persons']
            
        except Exception as e:
            logger.error(f"Fehler bei Bildanalyse {image_path}: {e}")
        
        return results
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Erkennt und identifiziert Gesichter im Bild"""
        faces = []
        
        if not self.known_face_encodings:
            return faces
        
        try:
            # RGB konvertieren für face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Bild verkleinern für schnellere Verarbeitung
            small_image = cv2.resize(rgb_image, (0, 0), fx=0.5, fy=0.5)
            
            # Gesichter finden (auf CPU - face_recognition nutzt dlib)
            face_locations = face_recognition.face_locations(small_image, model="hog")
            
            if not face_locations:
                return faces
            
            # Gesichts-Encodings erstellen
            face_encodings = face_recognition.face_encodings(small_image, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Koordinaten zurück skalieren
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                
                # Gesicht mit bekannten vergleichen
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding,
                    tolerance=0.6
                )
                
                name = "Unknown"
                confidence = 0.0
                
                if True in matches:
                    # Face distances berechnen
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, 
                        face_encoding
                    )
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                
                faces.append({
                    'name': name,
                    'confidence': float(confidence),
                    'bbox': {
                        'x1': int(left),
                        'y1': int(top),
                        'x2': int(right),
                        'y2': int(bottom)
                    }
                })
                
                logger.debug(f"Gesicht erkannt: {name} (Konfidenz: {confidence:.2f})")
        
        except Exception as e:
            logger.error(f"Fehler bei Gesichtserkennung: {e}")
        
        return faces
    
    def _detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """Erkennt Objekte mit YOLO - GPU-beschleunigt"""
        results = {
            'objects': [],
            'vehicles': [],
            'persons': 0
        }
        
        if self.yolo_model is None:
            return results
        
        try:
            # YOLO Inference mit expliziter GPU-Nutzung
            detections = self.yolo_model(
                image, 
                verbose=False,
                device=self.device,  # Explizit GPU verwenden
                half=True if self.device == 'cuda' else False  # FP16 auf GPU für Speed
            )[0]
            
            vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
            
            for detection in detections.boxes.data:
                x1, y1, x2, y2, conf, cls = detection
                class_id = int(cls)
                class_name = self.yolo_model.names[class_id]
                confidence = float(conf)
                
                obj_data = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    }
                }
                
                results['objects'].append(obj_data)
                
                # Spezielle Kategorien
                if class_name == 'person':
                    results['persons'] += 1
                elif class_name in vehicle_classes:
                    results['vehicles'].append(obj_data)
                
                logger.debug(f"Objekt erkannt: {class_name} (Konfidenz: {confidence:.2f})")
        
        except Exception as e:
            logger.error(f"Fehler bei YOLO-Detektion: {e}")
        
        return results
    
    def analyze_video(self, video_path: Path, sample_rate: int = 30) -> Dict[str, Any]:
        """
        Analysiert Video durch Sampling von Frames - GPU-beschleunigt
        
        Args:
            video_path: Pfad zum Video
            sample_rate: Analysiere jeden N-ten Frame
        """
        results = {
            'faces': [],
            'objects': [],
            'vehicles': [],
            'max_persons': 0,
            'total_frames': 0,
            'analyzed_frames': 0,
            'analysis_timestamp': datetime.now(),
            'gpu_used': self.device == 'cuda'
        }
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                logger.error(f"Video konnte nicht geöffnet werden: {video_path}")
                return results
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            results['total_frames'] = total_frames
            
            frame_count = 0
            analyzed_count = 0

            # Track best frame for complete object data with confidence scores
            unique_faces = set()
            unique_objects = set()  # Still track unique classes for logging
            unique_vehicles = set()
            best_frame_score = 0
            best_frame_results = None  # Store complete detections from best frame

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Nur jeden N-ten Frame analysieren
                if frame_count % sample_rate != 0:
                    continue

                analyzed_count += 1

                # Frame analysieren
                frame_results = self.analyze_image_array(frame)

                # Calculate frame score (faces + persons have higher priority)
                frame_score = len(frame_results['faces']) * 10 + frame_results['persons'] * 5 + len(frame_results['objects'])

                # Track best frame (for complete object data with confidence)
                if frame_score > best_frame_score:
                    best_frame_score = frame_score
                    # Store COMPLETE detections (with confidence and bbox) from best frame
                    best_frame_results = {
                        'objects': frame_results['objects'].copy(),
                        'vehicles': frame_results['vehicles'].copy(),
                        'persons': frame_results['persons']
                    }
                    logger.debug(f"New best frame: #{frame_count}, Score: {frame_score}")

                # Gesichter sammeln
                for face in frame_results['faces']:
                    if face['name'] != "Unknown":
                        unique_faces.add(face['name'])

                # Objekte sammeln (für Statistik/Logging)
                for obj in frame_results['objects']:
                    unique_objects.add(obj['class'])

                # Fahrzeuge sammeln (für Statistik/Logging)
                for vehicle in frame_results['vehicles']:
                    unique_vehicles.add(vehicle['class'])

                # Max Personen tracken
                if frame_results['persons'] > results['max_persons']:
                    results['max_persons'] = frame_results['persons']

                if analyzed_count % 10 == 0:
                    logger.debug(f"Video-Analyse: {analyzed_count} Frames analysiert")

            cap.release()

            results['analyzed_frames'] = analyzed_count
            results['faces'] = [{'name': name} for name in unique_faces]

            # Use detections from BEST frame (with confidence and bbox data!)
            if best_frame_results:
                results['objects'] = best_frame_results['objects']
                results['vehicles'] = best_frame_results['vehicles']
                logger.debug(f"Using {len(best_frame_results['objects'])} objects from best frame")
            else:
                # Fallback: No best frame found (empty video)
                # Return empty lists to avoid storing objects without confidence
                results['objects'] = []
                results['vehicles'] = []
                if unique_objects or unique_vehicles:
                    logger.warning(f"Video {video_path}: Objects detected ({unique_objects}) but no best frame found - data incomplete")
            
            logger.info(f"Video analysiert: {analyzed_count}/{total_frames} Frames, "
                       f"{len(unique_faces)} Gesichter, {results['max_persons']} max. Personen")

            # Debug: Log object confidence scores
            for i, obj in enumerate(results.get('objects', [])):
                logger.debug(f"  Object {i}: {obj.get('class')}, confidence={'confidence' in obj and obj['confidence'] or 'MISSING'}")
        
        except Exception as e:
            logger.error(f"Fehler bei Video-Analyse {video_path}: {e}")
        
        return results
    
    def analyze_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """Analysiert Bild als numpy array (für Video-Frames)"""
        results = {
            'faces': [],
            'objects': [],
            'vehicles': [],
            'persons': 0
        }
        
        try:
            # Gesichtserkennung
            face_results = self._detect_faces(image)
            results['faces'] = face_results
            
            # Objekt-Detektion
            yolo_results = self._detect_objects(image)
            results['objects'] = yolo_results['objects']
            results['vehicles'] = yolo_results['vehicles']
            results['persons'] = yolo_results['persons']
        
        except Exception as e:
            logger.error(f"Fehler bei Array-Analyse: {e}")
        
        return results


class FileProcessor:
    """Verarbeitet Mediendateien mit AI-Analyse und Datenbank-Integration"""
    
    def __init__(self, base_path: str, db_config: dict, ai_analyzer: AIAnalyzer):
        self.base_path = Path(base_path)
        self.db_config = db_config
        self.db_connection = None
        self.ai_analyzer = ai_analyzer
        
        # Statistiken
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.analyzed_count = 0
        self.total_analysis_time = 0.0
        
    def connect_db(self) -> bool:
        """Stellt Datenbankverbindung her"""
        try:
            self.db_connection = mysql.connector.connect(**self.db_config)
            if self.db_connection.is_connected():
                logger.info("Datenbankverbindung erfolgreich hergestellt")
                return True
        except MySQLError as e:
            logger.error(f"Datenbankverbindung fehlgeschlagen: {e}")
            return False
        return False
    
    def disconnect_db(self):
        """Schließt Datenbankverbindung"""
        if self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
            logger.info("Datenbankverbindung geschlossen")
    
    def parse_filename(self, filename: str) -> Optional[Tuple[str, str, datetime]]:
        """Parst Dateinamen nach dem Muster: Camera1_00_20260121074033.jpg/mp4"""
        pattern = r'^(Camera\d+)_(\d{2})_(\d{14})\.(jpg|mp4)$'
        match = re.match(pattern, filename)
        
        if not match:
            return None
        
        camera_name = match.group(1)
        timestamp_str = match.group(3)
        file_extension = match.group(4)
        
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
            return (camera_name, file_extension, timestamp)
        except ValueError as e:
            logger.error(f"Zeitstempel-Parsing fehlgeschlagen für {filename}: {e}")
            return None
    
    def file_exists_in_db(self, filepath: str) -> bool:
        """Prüft ob Datei bereits in DB existiert"""
        try:
            cursor = self.db_connection.cursor()
            query = "SELECT COUNT(*) FROM cam2_recordings WHERE file_path = %s"
            cursor.execute(query, (filepath,))
            count = cursor.fetchone()[0]
            cursor.close()
            return count > 0
        except MySQLError as e:
            logger.error(f"DB-Abfrage fehlgeschlagen: {e}")
            return False
    
    def insert_file_to_db(self, filepath: Path, camera_name: str, 
                         file_type: str, timestamp: datetime,
                         analysis_results: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """Trägt Datei mit AI-Analyse in die Datenbank ein"""
        try:
            cursor = self.db_connection.cursor()
            
            # Relativer Pfad zur Basis
            rel_path = str(filepath.relative_to(self.base_path))
            
            # Dateigröße ermitteln
            file_size = filepath.stat().st_size
            
            # Haupt-Recording eintragen
            query = """
                INSERT INTO cam2_recordings
                (camera_name, file_path, file_type, file_size, recorded_at,
                 analyzed, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """
            
            values = (
                camera_name,
                rel_path,
                file_type,
                file_size,
                timestamp,
                analysis_results is not None
            )
            
            cursor.execute(query, values)
            recording_id = cursor.lastrowid
            
            # AI-Analyse Ergebnisse eintragen
            if analysis_results:
                self._insert_analysis_results(cursor, recording_id, analysis_results)
            
            self.db_connection.commit()
            cursor.close()
            
            logger.info(f"✓ Eingetragen: {rel_path} (ID: {recording_id}, {file_size/1024/1024:.2f} MB)")
            return recording_id
            
        except MySQLError as e:
            logger.error(f"DB-Insert fehlgeschlagen für {filepath}: {e}")
            self.db_connection.rollback()
            return None
    
    def _insert_analysis_results(self, cursor, recording_id: int, results: Dict[str, Any]):
        """Trägt AI-Analyse Ergebnisse in DB ein"""
        try:
            # Gesichter eintragen
            for face in results.get('faces', []):
                query = """
                    INSERT INTO cam2_detected_faces
                    (recording_id, person_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                bbox = face.get('bbox', {})
                values = (
                    recording_id,
                    face.get('name', 'Unknown'),
                    face.get('confidence', 0.0),
                    bbox.get('x1', 0),
                    bbox.get('y1', 0),
                    bbox.get('x2', 0),
                    bbox.get('y2', 0)
                )
                cursor.execute(query, values)
            
            # Objekte eintragen
            for obj in results.get('objects', []):
                # Debug logging to track confidence values
                confidence = obj.get('confidence', 0.0)
                logger.debug(f"Inserting object: class={obj.get('class')}, confidence={confidence}, has_key={'confidence' in obj}")

                query = """
                    INSERT INTO cam2_detected_objects
                    (recording_id, object_class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                bbox = obj.get('bbox', {})
                values = (
                    recording_id,
                    obj.get('class', 'unknown'),
                    confidence,
                    bbox.get('x1', 0),
                    bbox.get('y1', 0),
                    bbox.get('x2', 0),
                    bbox.get('y2', 0)
                )
                cursor.execute(query, values)
            
            # Zusammenfassung eintragen
            query = """
                INSERT INTO cam2_analysis_summary
                (recording_id, total_faces, total_objects, total_vehicles,
                 max_persons, gpu_used, analyzed_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """
            values = (
                recording_id,
                len(results.get('faces', [])),
                len(results.get('objects', [])),
                len(results.get('vehicles', [])),
                results.get('max_persons', results.get('persons', 0)),
                results.get('gpu_used', False)
            )
            cursor.execute(query, values)
            
        except MySQLError as e:
            logger.error(f"Fehler beim Eintragen der Analyse-Ergebnisse: {e}")
            raise
    
    def find_all_media_files(self) -> List[Path]:
        """Findet alle Mediendateien rekursiv"""
        media_files = []
        
        for ext in ['*.mp4', '*.jpg']:
            media_files.extend(self.base_path.rglob(ext))
        
        # Nach Datum sortieren (älteste zuerst)
        media_files.sort(key=lambda p: p.stat().st_mtime)
        
        logger.info(f"Gefunden: {len(media_files)} Mediendateien")
        return media_files
    
    def process_file(self, filepath: Path, analyze: bool = True) -> bool:
        """Verarbeitet eine einzelne Datei mit optionaler AI-Analyse"""
        filename = filepath.name
        
        # Dateinamen parsen
        parsed = self.parse_filename(filename)
        if not parsed:
            logger.warning(f"⚠ Übersprungen (ungültiges Format): {filename}")
            self.skipped_count += 1
            return False
        
        camera_name, file_type, timestamp = parsed
        
        # Prüfen ob bereits in DB
        rel_path = str(filepath.relative_to(self.base_path))
        if self.file_exists_in_db(rel_path):
            logger.debug(f"⊘ Bereits in DB: {filename}")
            self.skipped_count += 1
            return False
        
        # AI-Analyse durchführen
        analysis_results = None
        if analyze:
            try:
                analysis_start = time.time()
                
                if file_type == 'jpg':
                    logger.info(f"🔍 Analysiere Bild: {filename}")
                    analysis_results = self.ai_analyzer.analyze_image(filepath)
                elif file_type == 'mp4':
                    logger.info(f"🎥 Analysiere Video: {filename}")
                    analysis_results = self.ai_analyzer.analyze_video(filepath, sample_rate=30)
                
                analysis_time = time.time() - analysis_start
                self.total_analysis_time += analysis_time
                self.analyzed_count += 1
                
                logger.info(f"  ⏱ Analyse-Zeit: {analysis_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Fehler bei AI-Analyse von {filename}: {e}")
        
        # In DB eintragen
        recording_id = self.insert_file_to_db(
            filepath, camera_name, file_type, timestamp, analysis_results
        )
        
        if recording_id:
            self.processed_count += 1
            return True
        else:
            self.error_count += 1
            return False
    
    def process_all_files(self, limit: Optional[int] = None, analyze: bool = True):
        """Verarbeitet alle gefundenen Dateien"""
        if not self.connect_db():
            logger.error("Abbruch: Keine Datenbankverbindung")
            return
        
        logger.info("=" * 70)
        logger.info("Starte Dateiverarbeitung mit AI-Analyse")
        logger.info(f"AI-Device: {self.ai_analyzer.device}")
        logger.info(f"Bekannte Gesichter: {len(self.ai_analyzer.known_face_names)}")
        
        # GPU Stats anzeigen
        gpu_stats = self.ai_analyzer.get_gpu_stats()
        if gpu_stats['available']:
            logger.info(f"GPU: {gpu_stats['name']}")
            logger.info(f"GPU Memory: {gpu_stats['memory_allocated_mb']:.2f} MB / {gpu_stats['memory_total_gb']:.2f} GB")
        
        logger.info("=" * 70)
        
        media_files = self.find_all_media_files()
        
        if limit:
            media_files = media_files[:limit]
            logger.info(f"Limitierung aktiv: Verarbeite nur erste {limit} Dateien")
        
        start_time = time.time()
        last_gpu_check = time.time()
        
        for idx, filepath in enumerate(media_files, 1):
            # Fortschritt alle 50 Dateien
            if idx % 50 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed if elapsed > 0 else 0
                logger.info(f"Fortschritt: {idx}/{len(media_files)} Dateien "
                          f"({rate:.2f} Dateien/Sek)")
                
                # GPU Stats alle 50 Dateien
                if gpu_stats['available']:
                    gpu_stats = self.ai_analyzer.get_gpu_stats()
                    logger.info(f"GPU Memory: {gpu_stats['memory_allocated_mb']:.2f} MB")
            
            self.process_file(filepath, analyze=analyze)
        
        elapsed = time.time() - start_time
        
        logger.info("=" * 70)
        logger.info("Verarbeitung abgeschlossen")
        logger.info(f"Verarbeitet: {self.processed_count} Dateien")
        logger.info(f"AI-Analysiert: {self.analyzed_count} Dateien")
        logger.info(f"Übersprungen: {self.skipped_count} Dateien")
        logger.info(f"Fehler: {self.error_count} Dateien")
        logger.info(f"Gesamt-Dauer: {elapsed:.2f} Sekunden ({elapsed/60:.2f} Minuten)")
        if self.processed_count > 0:
            logger.info(f"Durchschnitt: {elapsed/self.processed_count:.2f} Sek/Datei")
        if self.analyzed_count > 0:
            logger.info(f"Reine Analyse-Zeit: {self.total_analysis_time:.2f} Sekunden")
            logger.info(f"Durchschnitt Analyse: {self.total_analysis_time/self.analyzed_count:.2f} Sek/Datei")
        
        # Finale GPU Stats
        if gpu_stats['available']:
            gpu_stats = self.ai_analyzer.get_gpu_stats()
            logger.info(f"Finale GPU Memory: {gpu_stats['memory_allocated_mb']:.2f} MB")
        
        logger.info("=" * 70)
        
        self.disconnect_db()


def create_database_schema():
    """Erstellt die notwendigen Datenbanktabellen (cam2_* modern schema)"""
    schema = """
    CREATE TABLE IF NOT EXISTS cam2_recordings (
        id INT AUTO_INCREMENT PRIMARY KEY,
        camera_name VARCHAR(50) NOT NULL,
        file_path VARCHAR(255) NOT NULL UNIQUE,
        file_type ENUM('jpg', 'mp4') NOT NULL,
        file_size BIGINT NOT NULL,
        recorded_at DATETIME NOT NULL,
        analyzed BOOLEAN DEFAULT FALSE,
        created_at DATETIME NOT NULL,
        INDEX idx_camera (camera_name),
        INDEX idx_recorded (recorded_at),
        INDEX idx_analyzed (analyzed),
        INDEX idx_type (file_type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

    CREATE TABLE IF NOT EXISTS cam2_detected_faces (
        id INT AUTO_INCREMENT PRIMARY KEY,
        recording_id INT NOT NULL,
        person_name VARCHAR(100) NOT NULL,
        confidence FLOAT NOT NULL,
        bbox_x1 INT,
        bbox_y1 INT,
        bbox_x2 INT,
        bbox_y2 INT,
        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (recording_id) REFERENCES cam2_recordings(id) ON DELETE CASCADE,
        INDEX idx_person (person_name),
        INDEX idx_recording (recording_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

    CREATE TABLE IF NOT EXISTS cam2_detected_objects (
        id INT AUTO_INCREMENT PRIMARY KEY,
        recording_id INT NOT NULL,
        object_class VARCHAR(50) NOT NULL,
        confidence FLOAT NOT NULL,
        bbox_x1 INT,
        bbox_y1 INT,
        bbox_x2 INT,
        bbox_y2 INT,
        parking_spot_id INT DEFAULT NULL,
        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (recording_id) REFERENCES cam2_recordings(id) ON DELETE CASCADE,
        INDEX idx_class (object_class),
        INDEX idx_recording (recording_id),
        INDEX idx_parking_spot (parking_spot_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

    CREATE TABLE IF NOT EXISTS cam2_analysis_summary (
        id INT AUTO_INCREMENT PRIMARY KEY,
        recording_id INT NOT NULL UNIQUE,
        total_faces INT DEFAULT 0,
        total_objects INT DEFAULT 0,
        total_vehicles INT DEFAULT 0,
        max_persons INT DEFAULT 0,
        gpu_used BOOLEAN DEFAULT FALSE,
        scene_category VARCHAR(50) DEFAULT NULL,
        analyzed_at DATETIME NOT NULL,
        FOREIGN KEY (recording_id) REFERENCES cam2_recordings(id) ON DELETE CASCADE,
        INDEX idx_recording (recording_id),
        INDEX idx_scene (scene_category)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        cursor = connection.cursor()
        
        for statement in schema.split(';'):
            statement = statement.strip()
            if statement:
                cursor.execute(statement)
        
        connection.commit()
        cursor.close()
        connection.close()
        logger.info("✓ Datenbankschema erfolgreich erstellt/überprüft")
        return True
    except MySQLError as e:
        logger.error(f"Schema-Erstellung fehlgeschlagen: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='AI-Powered Video/Image Processor mit GPU-Beschleunigung'
    )
    parser.add_argument(
        '--base-path',
        default=MEDIA_BASE_PATH,
        help=f'Basis-Pfad für Mediendateien (default: {MEDIA_BASE_PATH})'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximale Anzahl zu verarbeitender Dateien (für Tests)'
    )
    parser.add_argument(
        '--no-analysis',
        action='store_true',
        help='Keine AI-Analyse durchführen (nur DB-Import)'
    )
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help='Erzwinge CPU-Nutzung (keine GPU)'
    )
    parser.add_argument(
        '--create-schema',
        action='store_true',
        help='Erstellt Datenbanktabellen falls nicht vorhanden'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Debug-Logging aktivieren'
    )
    parser.add_argument(
        '--yolo-model',
        default=YOLO_MODEL_PATH,
        help=f'Pfad zum YOLO-Model (default: {YOLO_MODEL_PATH})'
    )
    parser.add_argument(
        '--known-faces',
        default=KNOWN_FACES_DIR,
        help=f'Verzeichnis mit bekannten Gesichtern (default: {KNOWN_FACES_DIR})'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # GPU-Check durchführen
    gpu_available = check_gpu_availability()
    
    if not gpu_available and not args.cpu_only:
        logger.error("Keine GPU verfügbar! Verwende --cpu-only für CPU-Modus")
        sys.exit(1)
    
    if args.create_schema:
        logger.info("Erstelle Datenbankschema...")
        if not create_database_schema():
            sys.exit(1)
        logger.info("Schema erfolgreich erstellt")
    
    # AI Analyzer initialisieren
    logger.info("Initialisiere AI-Analyzer...")
    ai_analyzer = AIAnalyzer(
        args.yolo_model, 
        args.known_faces,
        force_gpu=not args.cpu_only
    )
    
    # Verarbeitung starten
    processor = FileProcessor(args.base_path, DB_CONFIG, ai_analyzer)
    processor.process_all_files(
        limit=args.limit,
        analyze=not args.no_analysis
    )


if __name__ == "__main__":
    main()
