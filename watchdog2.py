#!/home/gh/python/venv_py311/bin/python3
"""
AI Video Processor - GPU-Optimiert für Tesla P4
Verarbeitet vorhandene Video/Bild-Dateien mit vollständiger AI-Analyse
Gesichtserkennung mit Face-Cropping, Objekt-Detektion, Szenen-Erkennung

OPTIMIERT für Tesla P4 mit InsightFace GPU-Support
"""

import os
import sys

# KRITISCH: Environment-Variablen ZUERST setzen (vor allen AI-Imports!)
os.environ['ORT_DISABLE_CUDNN_FRONTEND'] = '1'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any
import re
import json

# Database imports
import pymysql

# AI/ML imports
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Configuration
MEDIA_BASE_PATH = "/var/www/web1"
ANNOTATED_OUTPUT_PATH = "/var/www/web1/annotated"
FACES_OUTPUT_PATH = "/var/www/web1/faces"
DB_CONFIG = {
    'host': 'localhost',  # TCP connection (not Unix socket)
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

# AI Model paths
YOLO_MODEL_PATH = "/opt/models/yolov8l.pt"  # Large model for better accuracy
KNOWN_FACES_DIR = "/opt/known_faces"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./logs/watchdog.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Überprüft GPU-Verfügbarkeit und zeigt Details - Tesla P4 Erkennung"""
    logger.info("=" * 70)
    logger.info("GPU STATUS CHECK")
    logger.info("=" * 70)
    
    # Python Environment Info
    logger.info(f"Python Executable: {sys.executable}")
    logger.info(f"Python Version: {sys.version}")
    
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
            
            # Tesla P4 Detection
            if "Tesla P4" in gpu_name or "P4" in gpu_name:
                logger.info("✓ Tesla P4 erkannt - optimale Einstellungen werden verwendet")
                logger.info("  - FP16 (Half Precision) aktiviert")
                logger.info("  - Batch-Inferenz optimiert")
                logger.info("  - CuDNN Frontend deaktiviert (für Kompatibilität)")
        
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
        logger.info(f"PyTorch Version: {torch.__version__}")
        
        # ONNX Runtime Check
        try:
            import onnxruntime as ort
            logger.info(f"ONNX Runtime Version: {ort.__version__}")
            providers = ort.get_available_providers()
            logger.info(f"ONNX Providers: {providers}")
            if 'CUDAExecutionProvider' in providers:
                logger.info("✓ ONNX Runtime GPU Support aktiviert")
            else:
                logger.warning("⚠ ONNX Runtime ohne GPU Support")
        except ImportError:
            logger.warning("⚠ ONNX Runtime nicht installiert")
            logger.info("Installation: pip install onnxruntime-gpu==1.17.1")
        
    else:
        logger.warning("⚠ CUDA nicht verfügbar! Läuft auf CPU.")
        logger.warning("Bitte PyTorch mit CUDA-Support installieren:")
        logger.warning("pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    logger.info("=" * 70)
    
    return cuda_available


def is_tesla_p4() -> bool:
    """Erkennt ob Tesla P4 GPU vorhanden ist"""
    if not torch.cuda.is_available():
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    return "Tesla P4" in gpu_name or "P4" in gpu_name


class FaceCropper:
    """
    Speichert erkannte Gesichter als separate Bild-Ausschnitte
    NEU: Nur Gesichter mit bekannten Namen (gelbe Boxen)
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FaceCropper initialisiert - Output: {self.output_dir}")
        
        self.saved_count = 0
    
    def save_face_crop(self, image: np.ndarray, face: Dict[str, Any], 
                      source_filename: str, recording_id: int) -> Optional[Path]:
        """
        Speichert Gesichts-Ausschnitt nur wenn Person bekannt ist
        
        Args:
            image: Original-Bild
            face: Face-Detection mit bbox und name
            source_filename: Original-Dateiname
            recording_id: Recording ID aus DB
            
        Returns:
            Pfad zum gespeicherten Crop oder None
        """
        # NUR bekannte Gesichter speichern
        if face.get('name') == 'Unknown':
            return None
        
        try:
            bbox = face['bbox']
            x1 = max(0, int(bbox['x1']))
            y1 = max(0, int(bbox['y1']))
            x2 = min(image.shape[1], int(bbox['x2']))
            y2 = min(image.shape[0], int(bbox['y2']))
            
            # Sicherheitsprüfung
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Ungültige Gesichts-Bbox: {bbox}")
                return None
            
            # Gesicht ausschneiden mit etwas Padding (10%)
            padding_x = int((x2 - x1) * 0.1)
            padding_y = int((y2 - y1) * 0.1)
            
            x1_padded = max(0, x1 - padding_x)
            y1_padded = max(0, y1 - padding_y)
            x2_padded = min(image.shape[1], x2 + padding_x)
            y2_padded = min(image.shape[0], y2 + padding_y)
            
            face_crop = image[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if face_crop.size == 0:
                logger.warning(f"Leerer Gesichts-Crop")
                return None
            
            # Dateiname generieren
            person_name = face['name'].replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            confidence = face.get('confidence', 0.0)
            
            filename = f"{person_name}_{timestamp}_conf{confidence:.2f}_rec{recording_id}.jpg"
            output_path = self.output_dir / filename
            
            # Speichern mit hoher Qualität
            cv2.imwrite(str(output_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            self.saved_count += 1
            logger.info(f"✓ Gesicht gespeichert: {filename} ({face_crop.shape[1]}x{face_crop.shape[0]}px)")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern von Gesichts-Crop: {e}")
            return None


class ImageAnnotator:
    """
    Zeichnet Bounding Boxes und Labels auf Bilder
    ANGEPASST: Gelbe Boxen für BEKANNTE Gesichter
    """
    
    # Farben (BGR Format für OpenCV)
    COLOR_VEHICLE = (0, 255, 255)   # Gelb
    COLOR_PERSON = (0, 255, 0)      # Grün
    COLOR_KNOWN_FACE = (0, 255, 255)  # Gelb für bekannte Gesichter
    COLOR_UNKNOWN_FACE = (0, 0, 255) # Rot für unbekannte Gesichter
    COLOR_OBJECT = (255, 0, 255)    # Magenta
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ImageAnnotator initialisiert - Output: {self.output_dir}")
    
    def annotate_image(self, image_path: Path, analysis_results: Dict[str, Any], 
                      save_prefix: str = "annotated") -> Optional[Path]:
        """
        Zeichnet alle Detektionen auf das Bild
        GELBE BOXEN nur für BEKANNTE Gesichter
        """
        try:
            # Bild laden
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Bild konnte nicht geladen werden: {image_path}")
                return None
            
            # Detektionen zeichnen
            detection_count = 0
            known_faces_count = 0
            
            # 1. Fahrzeuge (Gelbe Boxen)
            for vehicle in analysis_results.get('vehicles', []):
                self._draw_bbox(image, vehicle, self.COLOR_VEHICLE, vehicle['class'])
                detection_count += 1
            
            # 2. Personen (Grüne Boxen)
            for obj in analysis_results.get('objects', []):
                if obj['class'] == 'person':
                    self._draw_bbox(image, obj, self.COLOR_PERSON, 'Person')
                    detection_count += 1
            
            # 3. Gesichter - GELB für bekannt, ROT für unbekannt
            for face in analysis_results.get('faces', []):
                if face['name'] != 'Unknown':
                    # BEKANNTE Gesichter = GELBE BOX
                    self._draw_face_bbox(image, face, self.COLOR_KNOWN_FACE)
                    detection_count += 1
                    known_faces_count += 1
                else:
                    # Unbekannte Gesichter = Rote Box
                    self._draw_face_bbox(image, face, self.COLOR_UNKNOWN_FACE)
                    detection_count += 1
            
            # 4. Andere interessante Objekte (Magenta)
            interesting_objects = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 
                                 'elephant', 'bear', 'zebra', 'giraffe']
            for obj in analysis_results.get('objects', []):
                if obj['class'] in interesting_objects:
                    self._draw_bbox(image, obj, self.COLOR_OBJECT, obj['class'])
                    detection_count += 1
            
            # Nur speichern wenn Detektionen vorhanden
            if detection_count == 0:
                logger.debug(f"Keine Detektionen in {image_path.name} - überspringe Annotation")
                return None
            
            # Info-Text oben links
            info_text = f"Detektionen: {detection_count} | Bekannte Gesichter: {known_faces_count} | GPU: {analysis_results.get('gpu_used', False)}"
            cv2.putText(image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Legende
            legend_y = 60
            cv2.putText(image, "Gelb = Bekannte Person/Fahrzeug", (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(image, "Gruen = Person", (10, legend_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Ausgabepfad erstellen
            output_filename = f"{save_prefix}_{image_path.name}"
            output_path = self.output_dir / output_filename
            
            # Speichern
            cv2.imwrite(str(output_path), image)
            logger.debug(f"✓ Annotiert ({detection_count} Det., {known_faces_count} bekannte): {output_filename}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Fehler beim Annotieren von {image_path}: {e}")
            return None
    
    def _draw_bbox(self, image: np.ndarray, detection: Dict[str, Any], 
                   color: Tuple[int, int, int], label_text: str):
        """Zeichnet Bounding Box mit Label"""
        bbox = detection['bbox']
        x1 = int(bbox['x1'])
        y1 = int(bbox['y1'])
        x2 = int(bbox['x2'])
        y2 = int(bbox['y2'])
        conf = detection.get('confidence', 0.0)
        
        # Box zeichnen (dickere Linie)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Label mit Hintergrund
        label = f"{label_text} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Hintergrund für bessere Lesbarkeit
        cv2.rectangle(image, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        # Text
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def _draw_face_bbox(self, image: np.ndarray, face: Dict[str, Any], 
                       color: Tuple[int, int, int]):
        """Zeichnet Gesichts-Bounding Box mit Namen"""
        bbox = face['bbox']
        x1 = bbox['x1']
        y1 = bbox['y1']
        x2 = bbox['x2']
        y2 = bbox['y2']
        conf = face.get('confidence', 0.0)
        name = face.get('name', 'Unknown')
        
        # Box zeichnen (dicker für bekannte Gesichter)
        thickness = 4 if name != 'Unknown' else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Label mit Hintergrund
        label = f"{name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        cv2.rectangle(image, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), 
                     color, -1)
        
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


class AIAnalyzer:
    """AI-Analyse Engine für Video und Bilder - Optimiert für Tesla P4 mit InsightFace GPU"""
    
    def __init__(self, yolo_model_path: str, known_faces_dir: str, force_gpu: bool = True):
        self.yolo_model = None
        self.known_faces_dir = Path(known_faces_dir)
        self.known_face_encodings = []
        self.known_face_names = []
        self.force_gpu = force_gpu
        self.is_tesla_p4 = False
        self.face_app = None  # InsightFace App
        
        # GPU-Device bestimmen
        if torch.cuda.is_available() and force_gpu:
            self.device = 'cuda'
            self.gpu_id = 0
            torch.cuda.set_device(self.gpu_id)
            
            # Tesla P4 Erkennung
            self.is_tesla_p4 = is_tesla_p4()
            if self.is_tesla_p4:
                logger.info("✓ Tesla P4 erkannt - Performance-Optimierungen aktiviert")
        else:
            self.device = 'cpu'
            self.gpu_id = None
            if force_gpu:
                logger.error("GPU erzwungen, aber CUDA nicht verfügbar!")
                sys.exit(1)
        
        logger.info(f"AI-Analyzer initialisiert - Device: {self.device}")
        
        # YOLO Model laden
        self._load_yolo_model(yolo_model_path)
        
        # InsightFace GPU initialisieren
        if self.device == 'cuda':
            self._init_insightface_gpu()
        
        # Bekannte Gesichter laden
        self._load_known_faces()
        
        # GPU Warmup
        if self.device == 'cuda':
            self._gpu_warmup()
    
    def _init_insightface_gpu(self):
        """
        Initialisiert InsightFace mit GPU-Support für Tesla P4
        Environment-Variablen müssen bereits gesetzt sein!
        """
        try:
            from insightface.app import FaceAnalysis
            
            logger.info("Initialisiere InsightFace mit GPU-Support...")
            
            # Tesla P4 optimierte CUDA-Optionen
            cuda_options = {
                'device_id': 0,
                'cudnn_conv_algo_search': 'DEFAULT',
                'gpu_mem_limit': 6 * 1024 * 1024 * 1024,  # 6GB für P4
                'arena_extend_strategy': 'kSameAsRequested',
            }
            
            self.face_app = FaceAnalysis(
                name='buffalo_l',
                providers=[('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
            )
            
            # det_size für Tesla P4 optimiert (1280x1280 für 4K-Bilder)
            self.face_app.prepare(ctx_id=0, det_size=(1280, 1280))
            
            # Provider-Check
            providers = self.face_app.det_model.session.get_providers()
            logger.info(f"✓ InsightFace GPU initialisiert")
            logger.info(f"  Providers: {providers}")
            
            if 'CUDAExecutionProvider' in providers:
                logger.info("  ✓ GPU-Acceleration AKTIV")
            else:
                logger.warning("  ⚠ GPU-Acceleration NICHT aktiv - läuft auf CPU!")
            
        except Exception as e:
            logger.error(f"InsightFace GPU-Init fehlgeschlagen: {e}")
            logger.warning("Fallback: InsightFace wird nicht verwendet")
            self.face_app = None
    
    def _load_yolo_model(self, model_path: str):
        """Lädt YOLO Model explizit auf GPU mit Tesla P4 Optimierungen"""
        try:
            logger.info(f"Lade YOLO Model: {model_path}")
            self.yolo_model = YOLO(model_path)
            
            # Explizit auf GPU verschieben
            if self.device == 'cuda':
                self.yolo_model.to(self.device)
                logger.info(f"✓ YOLO Model auf GPU {self.gpu_id} geladen")
                
                if self.is_tesla_p4:
                    logger.info("Tesla P4 YOLO-Optimierungen:")
                    logger.info("  - FP16 (Half Precision) aktiviert")
                    logger.info("  - Optimale Batch-Größe: 4-8")
                
                logger.info(f"Model-Device: {next(self.yolo_model.model.parameters()).device}")
            else:
                logger.info("YOLO Model auf CPU geladen")
                
        except Exception as e:
            logger.error(f"YOLO Model konnte nicht geladen werden: {e}")
            raise
    
    def _gpu_warmup(self):
        """Warmup für GPU - Tesla P4 optimiert"""
        logger.info("GPU Warmup (Tesla P4 optimiert)...")
        try:
            # Dummy-Bild für Warmup
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # YOLO Warmup
            warmup_iterations = 3 if self.is_tesla_p4 else 1
            
            for i in range(warmup_iterations):
                _ = self.yolo_model(
                    dummy_image, 
                    verbose=False, 
                    device=self.device,
                    half=True  # FP16 für Tesla P4
                )
            
            # InsightFace Warmup
            if self.face_app is not None:
                for i in range(2):
                    _ = self.face_app.get(dummy_image)
            
            logger.info(f"✓ GPU Warmup abgeschlossen ({warmup_iterations} YOLO + 2 InsightFace Iterationen)")
            
            # Memory nach Warmup
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**2
                logger.info(f"GPU Memory nach Warmup: {allocated:.2f} MB")
                
        except Exception as e:
            logger.error(f"GPU Warmup fehlgeschlagen: {e}")
    
    def _load_known_faces(self):
        """Lädt bekannte Gesichter - InsightFace GPU-basiert"""
        if not self.known_faces_dir.exists():
            logger.warning(f"Verzeichnis für bekannte Gesichter nicht gefunden: {self.known_faces_dir}")
            logger.warning(f"Erstelle Verzeichnis: {self.known_faces_dir}")
            self.known_faces_dir.mkdir(parents=True, exist_ok=True)
            return
        
        face_files = list(self.known_faces_dir.glob("*.jpg")) + list(self.known_faces_dir.glob("*.png"))
        
        if not face_files:
            logger.warning(f"Keine Gesichts-Bilder in {self.known_faces_dir} gefunden")
            logger.info("Um Gesichtserkennung zu aktivieren, legen Sie .jpg/.png Dateien in diesem Verzeichnis ab")
            logger.info("Dateiname = Name der Person (z.B. 'Max_Mustermann.jpg')")
            return
        
        logger.info(f"Lade {len(face_files)} bekannte Gesichter...")
        
        # InsightFace GPU-basiert laden (wenn verfügbar)
        if self.face_app is not None:
            for image_path in face_files:
                try:
                    img = cv2.imread(str(image_path))
                    if img is None:
                        logger.warning(f"  ⚠ Bild konnte nicht geladen werden: {image_path.name}")
                        continue
                    
                    faces = self.face_app.get(img)
                    
                    if faces:
                        # Erstes Gesicht nehmen und Embedding speichern
                        self.known_face_encodings.append(faces[0].embedding)
                        name = image_path.stem.replace('_', ' ')
                        self.known_face_names.append(name)
                        logger.info(f"  ✓ {name} (GPU)")
                    else:
                        logger.warning(f"  ⚠ Kein Gesicht gefunden: {image_path.name}")
                except Exception as e:
                    logger.error(f"  ✗ Fehler bei {image_path.name}: {e}")
        else:
            logger.warning("InsightFace nicht verfügbar - Gesichtserkennung deaktiviert")
            return
        
        logger.info(f"✓ {len(self.known_face_names)} bekannte Gesichter geladen (GPU-basiert)")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Gibt aktuelle GPU-Statistiken zurück"""
        if not torch.cuda.is_available():
            return {'available': False}
        
        return {
            'available': True,
            'device': self.gpu_id,
            'name': torch.cuda.get_device_name(self.gpu_id),
            'is_tesla_p4': self.is_tesla_p4,
            'memory_allocated_mb': torch.cuda.memory_allocated(self.gpu_id) / 1024**2,
            'memory_reserved_mb': torch.cuda.memory_reserved(self.gpu_id) / 1024**2,
            'memory_total_gb': torch.cuda.get_device_properties(self.gpu_id).total_memory / 1024**3
        }
    
    def analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Analysiert ein einzelnes Bild mit zweistufiger Erkennung:
        1. YOLO Objekt-Detektion (schnell)
        2. Gesichtserkennung NUR wenn YOLO Personen erkannt hat
        """
        results = {
            'faces': [],
            'objects': [],
            'vehicles': [],
            'persons': 0,
            'known_faces_count': 0,
            'analysis_timestamp': datetime.now(),
            'gpu_used': self.device == 'cuda'
        }

        try:
            # Bild laden
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Bild konnte nicht geladen werden: {image_path}")
                return results

            # STUFE 1: Objekt-Detektion mit YOLO (GPU) - schneller Pre-Filter
            yolo_results = self._detect_objects(image)
            results['objects'] = yolo_results['objects']
            results['vehicles'] = yolo_results['vehicles']
            results['persons'] = yolo_results['persons']

            # STUFE 2: Gesichtserkennung NUR wenn YOLO Personen gefunden hat
            if results['persons'] > 0:
                logger.debug(f"YOLO hat {results['persons']} Person(en) erkannt → Starte Gesichtserkennung")
                face_results = self._detect_faces(image)
                results['faces'] = face_results
                results['known_faces_count'] = sum(1 for f in face_results if f['name'] != 'Unknown')
            else:
                logger.debug("YOLO hat keine Personen erkannt → Überspringe Gesichtserkennung")
                results['faces'] = []
                results['known_faces_count'] = 0

            # Szenen-Klassifikation
            results['scene_category'] = self._classify_scene(results)

        except Exception as e:
            logger.error(f"Fehler bei Bildanalyse {image_path}: {e}")

        return results
    
    def _detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Erkennt Gesichter mit InsightFace (GPU-beschleunigt)
        STRENGE FILTER gegen Falsch-Positive (z.B. gezeichnete Gesichter)
        """
        faces = []

        if self.face_app is None:
            logger.debug("InsightFace nicht verfügbar - keine Gesichtserkennung")
            return faces

        try:
            # InsightFace Detection (GPU)
            detected_faces = self.face_app.get(image)

            # Konfigurierbare Schwellwerte für strenge Filterung
            MIN_DET_SCORE = 0.8          # Detection Confidence (0.8 = sehr sicher)
            MIN_FACE_SIZE = 50           # Mindestgröße in Pixeln (Breite oder Höhe)
            MIN_SIMILARITY = 0.5         # Similarity für bekannte Gesichter (erhöht von 0.4)

            for face in detected_faces:
                # FILTER 1: Detection Score - wie sicher ist es ein Gesicht?
                det_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0
                if det_score < MIN_DET_SCORE:
                    logger.debug(f"⊘ Gesicht verworfen: Detection Score zu niedrig ({det_score:.3f} < {MIN_DET_SCORE})")
                    continue

                # FILTER 2: Gesichtsgröße - zu kleine Gesichter sind oft Artefakte
                bbox = face.bbox.astype(int)
                face_width = bbox[2] - bbox[0]
                face_height = bbox[3] - bbox[1]

                if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                    logger.debug(f"⊘ Gesicht verworfen: Zu klein ({face_width}x{face_height}px < {MIN_FACE_SIZE}px)")
                    continue

                # FILTER 3: Aspekt-Verhältnis - echte Gesichter haben ca. 0.7-1.3 Verhältnis
                aspect_ratio = face_width / max(face_height, 1)
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    logger.debug(f"⊘ Gesicht verworfen: Unnatürliches Aspekt-Verhältnis ({aspect_ratio:.2f})")
                    continue

                # Face Recognition mit bekannten Gesichtern vergleichen
                name = "Unknown"
                confidence = 0.0

                if self.known_face_encodings:
                    # InsightFace Embedding
                    face_embedding = face.embedding

                    # Cosine Similarity mit allen bekannten Gesichtern
                    similarities = []
                    for known_embedding in self.known_face_encodings:
                        # Cosine Similarity berechnen
                        similarity = np.dot(face_embedding, known_embedding) / (
                            np.linalg.norm(face_embedding) * np.linalg.norm(known_embedding)
                        )
                        similarities.append(similarity)

                    # Bestes Match finden
                    if similarities:
                        best_idx = np.argmax(similarities)
                        best_similarity = similarities[best_idx]

                        # STRENGER Threshold für Match (0.5 statt 0.4)
                        # Nur sehr sichere Matches werden akzeptiert
                        if best_similarity > MIN_SIMILARITY:
                            name = self.known_face_names[best_idx]
                            confidence = float(best_similarity)

                # Embedding für DB-Speicherung
                face_embedding = face.embedding

                faces.append({
                    'name': name,
                    'confidence': float(confidence) if name != "Unknown" else 0.0,
                    'bbox': {
                        'x1': int(bbox[0]),
                        'y1': int(bbox[1]),
                        'x2': int(bbox[2]),
                        'y2': int(bbox[3])
                    },
                    'embedding': face_embedding,  # 512-dimensional vector
                    'det_score': det_score  # Detection Score für Debugging
                })

                if name != "Unknown":
                    logger.debug(f"✓ Bekanntes Gesicht (GPU): {name} (sim={confidence:.2f}, det={det_score:.2f})")
                else:
                    logger.debug(f"? Unbekanntes Gesicht: {face_width}x{face_height}px (det={det_score:.2f})")

        except Exception as e:
            logger.error(f"Fehler bei InsightFace GPU-Detection: {e}")
            import traceback
            logger.debug(traceback.format_exc())

        return faces

    def _calculate_parking_spot_id(self, bbox: Dict[str, float], image_width: int = 4512, image_height: int = 2512,
                                   grid_cols: int = 4, grid_rows: int = 3) -> int:
        """
        Berechnet parking_spot_id basierend auf Fahrzeug-Position im Grid

        Args:
            bbox: Bounding Box mit x1, y1, x2, y2
            image_width: Bildbreite (Standard: 4K Reolink)
            image_height: Bildhöhe
            grid_cols: Anzahl Grid-Spalten (Standard: 4)
            grid_rows: Anzahl Grid-Zeilen (Standard: 3)

        Returns:
            parking_spot_id: 1-12 (bei 4x3 Grid)
        """
        # Mittelpunkt des Fahrzeugs berechnen
        center_x = (bbox['x1'] + bbox['x2']) / 2
        center_y = (bbox['y1'] + bbox['y2']) / 2

        # Grid-Cell berechnen
        cell_width = image_width / grid_cols
        cell_height = image_height / grid_rows

        col = min(int(center_x / cell_width), grid_cols - 1)
        row = min(int(center_y / cell_height), grid_rows - 1)

        # parking_spot_id: 1-basiert, von links nach rechts, oben nach unten
        # Row 0: IDs 1-4, Row 1: IDs 5-8, Row 2: IDs 9-12
        parking_spot_id = (row * grid_cols) + col + 1

        return parking_spot_id

    def _detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        """Erkennt Objekte mit YOLO - Tesla P4 optimiert"""
        results = {
            'objects': [],
            'vehicles': [],
            'persons': 0
        }
        
        if self.yolo_model is None:
            return results
        
        try:
            # YOLO Inference mit Tesla P4 Optimierungen
            detections = self.yolo_model(
                image, 
                verbose=False,
                device=self.device,
                half=True if (self.device == 'cuda' and self.is_tesla_p4) else False,  # FP16 für P4
                conf=0.25,  # Confidence Threshold
                iou=0.45    # NMS IOU Threshold
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
                    # Berechne parking_spot_id für Fahrzeuge
                    image_height, image_width = image.shape[:2]
                    parking_spot_id = self._calculate_parking_spot_id(
                        obj_data['bbox'],
                        image_width,
                        image_height
                    )
                    obj_data['parking_spot_id'] = parking_spot_id
                    results['vehicles'].append(obj_data)
                    logger.debug(f"Fahrzeug erkannt: {class_name} (Konfidenz: {confidence:.2f}, Parkplatz: {parking_spot_id})")
                else:
                    logger.debug(f"Objekt erkannt: {class_name} (Konfidenz: {confidence:.2f})")
        
        except Exception as e:
            logger.error(f"Fehler bei YOLO-Detektion: {e}")
        
        return results

    def _select_best_faces(self, all_faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Wählt das beste Gesicht pro Person aus allen gesammelten Frames

        Kriterien:
        - Bbox-Größe (größer = näher = schärfer)
        - Konfidenz (bei bekannten Gesichtern)

        Returns:
            Liste der besten Gesichter (ein pro eindeutiger Person)
        """
        if not all_faces:
            return []

        # Gruppiere Gesichter nach Person (Name + Bbox-Position für Unknown-Clustering)
        face_groups = {}

        for face in all_faces:
            name = face['name']

            # Für "Unknown" clustern wir nach räumlicher Nähe
            if name == "Unknown":
                # Nutze Bbox-Center für Clustering
                bbox = face['bbox']
                center_x = (bbox['x1'] + bbox['x2']) / 2
                center_y = (bbox['y1'] + bbox['y2']) / 2

                # Finde ob es bereits einen Unknown in der Nähe gibt (200px Radius)
                cluster_key = None
                for existing_key in face_groups.keys():
                    if existing_key.startswith("Unknown_"):
                        # Hole erstes Gesicht dieser Gruppe
                        first_face = face_groups[existing_key][0]
                        first_bbox = first_face['bbox']
                        first_center_x = (first_bbox['x1'] + first_bbox['x2']) / 2
                        first_center_y = (first_bbox['y1'] + first_bbox['y2']) / 2

                        # Prüfe Distanz
                        dist = np.sqrt((center_x - first_center_x)**2 + (center_y - first_center_y)**2)
                        if dist < 200:  # 200px Cluster-Radius
                            cluster_key = existing_key
                            break

                # Neuer Cluster wenn kein Match
                if cluster_key is None:
                    cluster_key = f"Unknown_{len([k for k in face_groups.keys() if k.startswith('Unknown_')])}"

                if cluster_key not in face_groups:
                    face_groups[cluster_key] = []
                face_groups[cluster_key].append(face)
            else:
                # Bekannte Personen nach Name gruppieren
                if name not in face_groups:
                    face_groups[name] = []
                face_groups[name].append(face)

        # Wähle bestes Gesicht pro Gruppe
        best_faces = []

        for group_name, faces in face_groups.items():
            # Berechne Score für jedes Gesicht
            scored_faces = []

            for face in faces:
                # Score-Berechnung:
                # 70% Gewicht: Bbox-Größe (größer = besser)
                # 30% Gewicht: Confidence (bei bekannten Gesichtern)

                bbox_score = face['bbox_area'] / 10000.0  # Normalisierung (ca. 100x100 = 1.0)
                conf_score = face.get('confidence', 0.0)

                # Kombinierter Score
                total_score = (0.7 * bbox_score) + (0.3 * conf_score)

                scored_faces.append((total_score, face))

            # Bestes Gesicht auswählen
            scored_faces.sort(key=lambda x: x[0], reverse=True)
            best_face = scored_faces[0][1]

            # Entferne temporäre Felder
            best_face_clean = {
                'name': best_face['name'],
                'confidence': best_face.get('confidence', 0.0),
                'bbox': best_face['bbox']
            }

            best_faces.append(best_face_clean)

            logger.debug(f"Beste Auswahl für '{group_name}': Frame {best_face['frame_number']}, "
                        f"Größe {best_face['bbox_area']:.0f}px², Konfidenz {best_face.get('confidence', 0):.2f}")

        return best_faces

    def analyze_video(self, video_path: Path, sample_rate: int = 30) -> Dict[str, Any]:
        """
        Analysiert Video durch Sampling von Frames - Tesla P4 optimiert
        Mit intelligenter Frame-Selektion für beste Gesichter
        """
        results = {
            'faces': [],
            'objects': [],
            'vehicles': [],
            'max_persons': 0,
            'total_frames': 0,
            'analyzed_frames': 0,
            'known_faces_count': 0,
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

            # Sammle ALLE Gesichter aus ALLEN Frames (für beste Auswahl)
            all_faces = []  # Liste aller gefundenen Gesichter mit Frame-Info
            unique_objects = set()
            unique_vehicles = set()

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

                # Gesichter sammeln (ALLE, mit Frame-Nummer)
                for face in frame_results['faces']:
                    face_copy = face.copy()
                    face_copy['frame_number'] = frame_count
                    # Berechne Bbox-Area für Qualitätsbewertung
                    bbox = face['bbox']
                    face_copy['bbox_area'] = (bbox['x2'] - bbox['x1']) * (bbox['y2'] - bbox['y1'])
                    all_faces.append(face_copy)

                # Objekte sammeln
                for obj in frame_results['objects']:
                    unique_objects.add(obj['class'])

                # Fahrzeuge sammeln
                for vehicle in frame_results['vehicles']:
                    unique_vehicles.add(vehicle['class'])

                # Max Personen tracken
                if frame_results['persons'] > results['max_persons']:
                    results['max_persons'] = frame_results['persons']

                if analyzed_count % 10 == 0:
                    logger.debug(f"Video-Analyse: {analyzed_count} Frames analysiert")

            cap.release()

            # Wähle beste Gesichter aus (ein pro Person)
            best_faces = self._select_best_faces(all_faces)

            results['analyzed_frames'] = analyzed_count
            results['faces'] = best_faces
            results['known_faces_count'] = sum(1 for f in best_faces if f['name'] != 'Unknown')
            results['objects'] = [{'class': obj} for obj in unique_objects]
            results['vehicles'] = [{'class': veh} for veh in unique_vehicles]

            # Szenen-Klassifikation basierend auf Video-Analyse
            results['scene_category'] = self._classify_scene(results)

            logger.info(f"Video analysiert: {analyzed_count}/{total_frames} Frames, "
                       f"{results['known_faces_count']} bekannte Gesichter, {results['max_persons']} max. Personen, "
                       f"Szene: {results['scene_category']}")

        except Exception as e:
            logger.error(f"Fehler bei Video-Analyse {video_path}: {e}")

        return results

    def _classify_scene(self, results: Dict[str, Any]) -> str:
        """
        Klassifiziert die Szene basierend auf erkannten Objekten
        Kategorien: parking, street, entrance, indoor, empty
        """
        vehicles = results.get('vehicles', [])
        persons = results.get('persons', 0)
        objects = results.get('objects', [])

        # Zähle Objekt-Typen
        num_vehicles = len(vehicles)
        num_persons = persons

        # Klassifikationslogik
        if num_vehicles >= 2 and num_persons == 0:
            return 'parking'  # Mehrere Fahrzeuge, keine Personen = Parkplatz
        elif num_vehicles >= 1 and num_persons >= 1:
            return 'street'  # Fahrzeuge + Personen = Straßenszene
        elif num_persons >= 2 and num_vehicles == 0:
            return 'entrance'  # Mehrere Personen ohne Fahrzeuge = Eingangsbereich
        elif num_persons == 1 and num_vehicles == 0:
            return 'visitor'  # Eine Person = Besucher
        elif num_vehicles == 1 and num_persons == 0:
            return 'delivery'  # Ein Fahrzeug alleine = Lieferung/Anlieferung
        elif len(objects) == 0:
            return 'empty'  # Keine Objekte = leere Szene
        else:
            return 'unknown'  # Alles andere

    def analyze_image_array(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analysiert Bild als numpy array (für Video-Frames)
        Mit zweistufiger Erkennung: YOLO first, Faces nur wenn Person erkannt
        """
        results = {
            'faces': [],
            'objects': [],
            'vehicles': [],
            'persons': 0,
            'known_faces_count': 0
        }

        try:
            # STUFE 1: Objekt-Detektion mit YOLO (schneller Pre-Filter)
            yolo_results = self._detect_objects(image)
            results['objects'] = yolo_results['objects']
            results['vehicles'] = yolo_results['vehicles']
            results['persons'] = yolo_results['persons']

            # STUFE 2: Gesichtserkennung NUR wenn YOLO Personen gefunden hat
            if results['persons'] > 0:
                face_results = self._detect_faces(image)
                results['faces'] = face_results
                results['known_faces_count'] = sum(1 for f in face_results if f['name'] != 'Unknown')
            else:
                results['faces'] = []
                results['known_faces_count'] = 0

            # Szenen-Klassifikation
            results['scene_category'] = self._classify_scene(results)

        except Exception as e:
            logger.error(f"Fehler bei Array-Analyse: {e}")

        return results


class FileProcessor:
    """Verarbeitet Mediendateien mit AI-Analyse und Datenbank-Integration"""
    
    def __init__(self, base_path: str, db_config: dict, ai_analyzer: AIAnalyzer, 
                 annotator: Optional[ImageAnnotator] = None,
                 face_cropper: Optional[FaceCropper] = None,
                 reanalyze: bool = False):  # NEU
        self.base_path = Path(base_path)
        self.db_config = db_config
        self.db_connection = None
        self.ai_analyzer = ai_analyzer
        self.annotator = annotator
        self.face_cropper = face_cropper
        self.reanalyze = reanalyze  # NEU
        
        # Statistiken
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        self.analyzed_count = 0
        self.annotated_count = 0
        self.face_crops_saved = 0
        self.total_analysis_time = 0.0
        
    def connect_db(self) -> bool:
        """Stellt Datenbankverbindung her"""
        try:
            self.db_connection = pymysql.connect(**self.db_config)
            logger.info("Datenbankverbindung erfolgreich hergestellt")
            return True
        except Exception as e:
            logger.error(f"Datenbankverbindung fehlgeschlagen: {e}")
            return False
    
    def disconnect_db(self):
        """Schließt Datenbankverbindung"""
        if self.db_connection:
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
        except Exception as e:
            logger.error(f"DB-Abfrage fehlgeschlagen: {e}")
            return False
    
    def _delete_file_from_db(self, filepath: str):
        """Löscht Datei und alle zugehörigen Analysen aus DB (für --reanalyze)"""
        try:
            cursor = self.db_connection.cursor()
            
            # Hole recording_id
            query = "SELECT id FROM cam2_recordings WHERE file_path = %s"
            cursor.execute(query, (filepath,))
            result = cursor.fetchone()
            
            if result:
                recording_id = result[0]
                
                # Lösche Analysen (CASCADE löscht automatisch faces, objects, summary)
                query = "DELETE FROM cam2_recordings WHERE id = %s"
                cursor.execute(query, (recording_id,))
                
                self.db_connection.commit()
                logger.debug(f"✓ Alte DB-Einträge gelöscht: {filepath}")
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Fehler beim Löschen von DB-Einträgen für {filepath}: {e}")
            self.db_connection.rollback()
    
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
            
        except Exception as e:
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
                    (recording_id, person_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                bbox = face.get('bbox', {})

                # Face Embedding als Binary speichern (512-dim float32 = 2KB)
                embedding_bytes = None
                if 'embedding' in face and face['embedding'] is not None:
                    try:
                        embedding_bytes = face['embedding'].tobytes()
                    except Exception as e:
                        logger.warning(f"Fehler beim Konvertieren des Embeddings: {e}")

                values = (
                    recording_id,
                    face.get('name', 'Unknown'),
                    face.get('confidence', 0.0),
                    bbox.get('x1', 0),
                    bbox.get('y1', 0),
                    bbox.get('x2', 0),
                    bbox.get('y2', 0),
                    embedding_bytes
                )
                cursor.execute(query, values)
            
            # Objekte eintragen
            for obj in results.get('objects', []):
                query = """
                    INSERT INTO cam2_detected_objects
                    (recording_id, object_class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, parking_spot_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                bbox = obj.get('bbox', {})
                values = (
                    recording_id,
                    obj.get('class', 'unknown'),
                    obj.get('confidence', 0.0),
                    bbox.get('x1', 0),
                    bbox.get('y1', 0),
                    bbox.get('x2', 0),
                    bbox.get('y2', 0),
                    obj.get('parking_spot_id', None)  # NULL für nicht-Fahrzeuge
                )
                cursor.execute(query, values)
            
            # Zusammenfassung eintragen
            query = """
                INSERT INTO cam2_analysis_summary
                (recording_id, total_faces, total_objects, total_vehicles,
                 max_persons, scene_category, gpu_used, analyzed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """
            values = (
                recording_id,
                len(results.get('faces', [])),
                len(results.get('objects', [])),
                len(results.get('vehicles', [])),
                results.get('max_persons', results.get('persons', 0)),
                results.get('scene_category', 'unknown'),
                results.get('gpu_used', False)
            )
            cursor.execute(query, values)
            
        except Exception as e:
            logger.error(f"Fehler beim Eintragen der Analyse-Ergebnisse: {e}")
            raise

    def get_face_embedding(self, face_id: int) -> Optional[np.ndarray]:
        """
        Lädt Face Embedding aus der Datenbank

        Returns:
            512-dim numpy array oder None wenn kein Embedding vorhanden
        """
        try:
            cursor = self.db_connection.cursor()
            query = "SELECT embedding FROM cam2_detected_faces WHERE id = %s"
            cursor.execute(query, (face_id,))
            result = cursor.fetchone()
            cursor.close()

            if result and result[0]:
                # BLOB zurück zu numpy array konvertieren
                embedding = np.frombuffer(result[0], dtype=np.float32)
                if len(embedding) == 512:
                    return embedding
                else:
                    logger.warning(f"Ungültige Embedding-Größe für face_id {face_id}: {len(embedding)}")
                    return None
            return None

        except Exception as e:
            logger.error(f"Fehler beim Laden des Embeddings für face_id {face_id}: {e}")
            return None

    def calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Berechnet Cosine Similarity zwischen zwei Embeddings

        Returns:
            Float zwischen 0.0 und 1.0 (1.0 = identisch)
        """
        # Normalisiere Vektoren
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine Similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Clamp auf [0, 1]
        return float(max(0.0, min(1.0, similarity)))

    def find_similar_faces(self, face_id: int, threshold: float = 0.6, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Findet ähnliche Gesichter basierend auf Embedding Similarity

        Args:
            face_id: ID des Referenz-Gesichts
            threshold: Minimum Cosine Similarity (0.6 = gute Balance)
            limit: Maximale Anzahl Ergebnisse

        Returns:
            Liste von Dicts mit face_id, similarity, person_name, recording_id
        """
        # Lade Referenz-Embedding
        ref_embedding = self.get_face_embedding(face_id)
        if ref_embedding is None:
            logger.warning(f"Kein Embedding für face_id {face_id}")
            return []

        try:
            cursor = self.db_connection.cursor(pymysql.cursors.DictCursor)

            # Lade alle Gesichter mit Embeddings (außer das Referenz-Gesicht selbst)
            query = """
                SELECT id, person_name, confidence, recording_id, embedding
                FROM cam2_detected_faces
                WHERE id != %s AND embedding IS NOT NULL
                ORDER BY id DESC
                LIMIT 1000
            """
            cursor.execute(query, (face_id,))
            faces = cursor.fetchall()
            cursor.close()

            # Berechne Similarities
            results = []
            for face in faces:
                if face['embedding']:
                    embedding = np.frombuffer(face['embedding'], dtype=np.float32)
                    if len(embedding) == 512:
                        similarity = self.calculate_cosine_similarity(ref_embedding, embedding)

                        if similarity >= threshold:
                            results.append({
                                'face_id': face['id'],
                                'similarity': similarity,
                                'person_name': face['person_name'],
                                'confidence': face['confidence'],
                                'recording_id': face['recording_id']
                            })

            # Sortiere nach Similarity (beste zuerst)
            results.sort(key=lambda x: x['similarity'], reverse=True)

            return results[:limit]

        except Exception as e:
            logger.error(f"Fehler bei Similarity-Suche: {e}")
            return []

    def bulk_label_similar_faces(self, face_id: int, new_name: str, threshold: float = 0.65,
                                 dry_run: bool = True) -> Dict[str, Any]:
        """
        Benennt alle ähnlichen Gesichter um (Bulk-Labeling)

        Args:
            face_id: Referenz-Gesicht
            new_name: Neuer Name für alle ähnlichen Gesichter
            threshold: Similarity-Schwellwert (0.65 = konservativ)
            dry_run: Wenn True, nur Simulation ohne DB-Update

        Returns:
            Dict mit 'count', 'face_ids', 'dry_run' Status
        """
        similar_faces = self.find_similar_faces(face_id, threshold=threshold, limit=1000)

        if not similar_faces:
            logger.info(f"Keine ähnlichen Gesichter gefunden (threshold={threshold})")
            return {'count': 0, 'face_ids': [], 'dry_run': dry_run}

        face_ids = [f['face_id'] for f in similar_faces]

        if dry_run:
            logger.info(f"[DRY RUN] Würde {len(face_ids)} Gesichter umbenennen zu '{new_name}'")
            return {'count': len(face_ids), 'face_ids': face_ids, 'dry_run': True}

        # Echtes Update
        try:
            cursor = self.db_connection.cursor()

            # Batch-Update
            placeholders = ', '.join(['%s'] * len(face_ids))
            query = f"""
                UPDATE cam2_detected_faces
                SET person_name = %s
                WHERE id IN ({placeholders})
            """

            cursor.execute(query, [new_name] + face_ids)
            self.db_connection.commit()
            cursor.close()

            logger.info(f"✓ {len(face_ids)} Gesichter umbenannt zu '{new_name}'")
            return {'count': len(face_ids), 'face_ids': face_ids, 'dry_run': False}

        except Exception as e:
            logger.error(f"Fehler beim Bulk-Labeling: {e}")
            self.db_connection.rollback()
            return {'count': 0, 'face_ids': [], 'dry_run': False, 'error': str(e)}

    def find_all_media_files(self) -> List[Path]:
        """Findet alle Mediendateien rekursiv"""
        media_files = []
        
        for ext in ['*.mp4', '*.jpg']:
            media_files.extend(self.base_path.rglob(ext))
        
        # Nach Datum sortieren (neueste zuerst)
        media_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
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
        
        # NEU: Bei --reanalyze alte Einträge löschen
        if self.reanalyze and self.file_exists_in_db(rel_path):
            self._delete_file_from_db(rel_path)
            logger.info(f"🔄 Reanalyze: Alte Daten gelöscht für {filename}")
        elif self.file_exists_in_db(rel_path):
            logger.debug(f"⊘ Bereits in DB: {filename}")
            self.skipped_count += 1
            return False
        
        # AI-Analyse durchführen
        analysis_results = None
        recording_id = None
        
        if analyze:
            try:
                analysis_start = time.time()
                
                if file_type == 'jpg':
                    logger.info(f"🔍 Analysiere Bild: {filename}")
                    analysis_results = self.ai_analyzer.analyze_image(filepath)
                    
                    # Zuerst in DB eintragen um recording_id zu bekommen
                    recording_id = self.insert_file_to_db(
                        filepath, camera_name, file_type, timestamp, analysis_results
                    )
                    
                    if recording_id:
                        # Face-Crops speichern (nur für bekannte Gesichter)
                        if self.face_cropper and analysis_results.get('known_faces_count', 0) > 0:
                            image = cv2.imread(str(filepath))
                            if image is not None:
                                for face in analysis_results.get('faces', []):
                                    if face['name'] != 'Unknown':
                                        crop_path = self.face_cropper.save_face_crop(
                                            image, face, filename, recording_id
                                        )
                                        if crop_path:
                                            self.face_crops_saved += 1
                        
                        # Optional: Annotiertes Bild erstellen
                        if self.annotator and analysis_results:
                            annotated_path = self.annotator.annotate_image(
                                filepath, analysis_results
                            )
                            if annotated_path:
                                self.annotated_count += 1
                    
                elif file_type == 'mp4':
                    logger.info(f"🎥 Analysiere Video: {filename}")
                    analysis_results = self.ai_analyzer.analyze_video(filepath, sample_rate=1)
                    
                    # In DB eintragen
                    recording_id = self.insert_file_to_db(
                        filepath, camera_name, file_type, timestamp, analysis_results
                    )
                
                analysis_time = time.time() - analysis_start
                self.total_analysis_time += analysis_time
                self.analyzed_count += 1
                
                logger.info(f"  ⏱ Analyse-Zeit: {analysis_time:.2f}s")
                
                if recording_id:
                    self.processed_count += 1
                    return True
                
            except Exception as e:
                logger.error(f"Fehler bei AI-Analyse von {filename}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                self.error_count += 1
                return False
        else:
            # Ohne Analyse nur in DB eintragen
            recording_id = self.insert_file_to_db(
                filepath, camera_name, file_type, timestamp, None
            )
            
            if recording_id:
                self.processed_count += 1
                return True
        
        self.error_count += 1
        return False
    
    def process_all_files(self, limit: Optional[int] = None, analyze: bool = True):
        """Verarbeitet alle gefundenen Dateien"""
        if not self.connect_db():
            logger.error("Abbruch: Keine Datenbankverbindung")
            return
        
        logger.info("=" * 70)
        logger.info("Starte Dateiverarbeitung mit AI-Analyse (Tesla P4 optimiert)")
        logger.info(f"AI-Device: {self.ai_analyzer.device}")
        logger.info(f"Bekannte Gesichter: {len(self.ai_analyzer.known_face_names)}")
        logger.info(f"Annotation: {'Aktiviert' if self.annotator else 'Deaktiviert'}")
        logger.info(f"Face-Cropping: {'Aktiviert (nur bekannte)' if self.face_cropper else 'Deaktiviert'}")
        logger.info(f"InsightFace GPU: {'Aktiviert' if self.ai_analyzer.face_app else 'Deaktiviert'}")
        logger.info(f"Reanalyze-Modus: {'Aktiviert (überschreibt bestehende Analysen)' if self.reanalyze else 'Deaktiviert'}")  # NEU
        
        # GPU Stats anzeigen
        gpu_stats = self.ai_analyzer.get_gpu_stats()
        if gpu_stats['available']:
            logger.info(f"GPU: {gpu_stats['name']}")
            if gpu_stats.get('is_tesla_p4'):
                logger.info("✓ Tesla P4 erkannt - optimale Performance-Einstellungen aktiv")
            logger.info(f"GPU Memory: {gpu_stats['memory_allocated_mb']:.2f} MB / {gpu_stats['memory_total_gb']:.2f} GB")
        
        logger.info("=" * 70)
        
        media_files = self.find_all_media_files()
        
        if limit:
            media_files = media_files[:limit]
            logger.info(f"Limitierung aktiv: Verarbeite nur erste {limit} Dateien")
        
        start_time = time.time()
        
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
        logger.info(f"Annotiert: {self.annotated_count} Bilder")
        logger.info(f"Gesichts-Crops: {self.face_crops_saved} (nur bekannte Personen)")
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
    """Erstellt die notwendigen Datenbanktabellen"""
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
        embedding BLOB,
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
        scene_category VARCHAR(50) DEFAULT 'unknown',
        gpu_used BOOLEAN DEFAULT FALSE,
        analyzed_at DATETIME NOT NULL,
        FOREIGN KEY (recording_id) REFERENCES cam2_recordings(id) ON DELETE CASCADE,
        INDEX idx_recording (recording_id),
        INDEX idx_scene (scene_category)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

    CREATE TABLE IF NOT EXISTS cam2_parking_stats (
        id INT AUTO_INCREMENT PRIMARY KEY,
        parking_spot_id INT NOT NULL,
        date DATE NOT NULL,
        hour INT NOT NULL,
        occupancy_count INT DEFAULT 0,
        vehicle_type VARCHAR(50),
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY unique_spot_time (parking_spot_id, date, hour, vehicle_type),
        INDEX idx_spot (parking_spot_id),
        INDEX idx_date (date),
        INDEX idx_spot_date (parking_spot_id, date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    try:
        connection = pymysql.connect(**DB_CONFIG)
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
    except Exception as e:
        logger.error(f"Schema-Erstellung fehlgeschlagen: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='AI-Powered Video/Image Processor - Tesla P4 Optimiert mit InsightFace GPU'
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
        '--save-annotated',
        action='store_true',
        help='Speichert annotierte Bilder mit Bounding Boxes (gelb = bekannt)'
    )
    parser.add_argument(
        '--save-face-crops',
        action='store_true',
        help='Speichert Gesichts-Crops (nur bekannte Personen)'
    )
    parser.add_argument(
        '--annotated-path',
        default=ANNOTATED_OUTPUT_PATH,
        help=f'Ausgabepfad für annotierte Bilder (default: {ANNOTATED_OUTPUT_PATH})'
    )
    parser.add_argument(
        '--faces-path',
        default=FACES_OUTPUT_PATH,
        help=f'Ausgabepfad für Gesichts-Crops (default: {FACES_OUTPUT_PATH})'
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
    parser.add_argument(
        '--reanalyze',
        action='store_true',
        help='Analysiert bereits analysierte Dateien erneut (überschreibt DB-Einträge)'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # GPU-Check durchführen (mit Tesla P4 Erkennung)
    gpu_available = check_gpu_availability()
    
    if not gpu_available and not args.cpu_only:
        logger.error("Keine GPU verfügbar! Verwende --cpu-only für CPU-Modus")
        logger.error("Oder installiere PyTorch mit CUDA:")
        logger.error("  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        sys.exit(1)
    
    if args.create_schema:
        logger.info("Erstelle Datenbankschema...")
        if not create_database_schema():
            sys.exit(1)
        logger.info("Schema erfolgreich erstellt")
    
    # AI Analyzer initialisieren
    logger.info("Initialisiere AI-Analyzer (Tesla P4 + InsightFace GPU)...")
    ai_analyzer = AIAnalyzer(
        args.yolo_model, 
        args.known_faces,
        force_gpu=not args.cpu_only
    )
    
    # Optional: Annotator initialisieren
    annotator = None
    if args.save_annotated:
        logger.info("Annotation aktiviert (gelbe Boxen = bekannte Personen)...")
        annotator = ImageAnnotator(args.annotated_path)
    
    # Optional: Face-Cropper initialisieren
    face_cropper = None
    if args.save_face_crops:
        logger.info("Face-Cropping aktiviert (nur bekannte Personen)...")
        face_cropper = FaceCropper(args.faces_path)
    
    # Verarbeitung starten
    processor = FileProcessor(
        args.base_path, 
        DB_CONFIG, 
        ai_analyzer, 
        annotator,
        face_cropper,
        reanalyze=args.reanalyze  # NEU
    )
    processor.process_all_files(
        limit=args.limit,
        analyze=not args.no_analysis
    )


if __name__ == "__main__":
    main()
