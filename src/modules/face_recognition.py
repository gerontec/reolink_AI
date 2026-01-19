"""
Face Recognition Module - Simplified
"""
import torch
import pickle
import os
import cv2
import numpy as np

class FeatureModule:
    def __init__(self, config):
        self.config = config.get('face_recognition', {})
        self.threshold = self.config.get('threshold', 0.6)
        self.skip_alarm = self.config.get('skip_alarm_for_known', True)
        
        print("  Loading face models...")
        
        try:
            from facenet_pytorch import MTCNN, InceptionResnetV1
            
            self.detector = MTCNN(image_size=160, margin=20, device='cuda')
            self.recognizer = InceptionResnetV1(pretrained='vbggface2').to('cuda').eval()
            
            # Datenbank laden
            db_path = self.config.get('database', 'face_gpu.pkl')
            if os.path.exists(db_path):
                with open(db_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
                    if isinstance(self.known_faces, list):
                        self.known_faces = {'enrolled': self.known_faces}
                print(f"  Loaded {len(self.known_faces)} face(s)")
            else:
                self.known_faces = {}
                print("  No faces enrolled yet")
            
            self.enabled = True
            
        except Exception as e:
            print(f"  Face recognition disabled: {e}")
            self.enabled = False
    
    def process(self, frame, detection_data):
        """Erkennt Gesichter"""
        if not self.enabled:
            return None
            
        if detection_data['object_type'] != 'person':
            return None
        
        if not detection_data.get('in_upper_zone', False):
            return None
        
        try:
            x1, y1, x2, y2 = map(int, detection_data['bbox'])
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.shape[0] < 50 or crop.shape[1] < 50:
                return None
            
            # Detect face
            boxes, probs = self.detector.detect(crop)
            
            if boxes is None:
                return {'recognized': False, 'reason': 'no_face'}
            
            # Get embedding
            face_img = self.detector(crop)
            if face_img is None:
                return {'recognized': False, 'reason': 'no_face'}
            
            with torch.no_grad():
                embedding = self.recognizer(face_img.unsqueeze(0).to('cuda'))
            
            # Compare
            best_dist = 999.0
            best_name = None
            
            for name, known_embs in self.known_faces.items():
                if not isinstance(known_embs, list):
                    known_embs = [known_embs]
                
                for known_emb in known_embs:
                    if isinstance(known_emb, torch.Tensor):
                        dist = torch.dist(embedding, known_emb.to('cuda')).item()
                        if dist < best_dist:
                            best_dist = dist
                            best_name = name
            
            recognized = best_dist < self.threshold
            
            return {
                'recognized': recognized,
                'name': best_name if recognized else 'unknown',
                'distance': float(best_dist),
                'skip_alarm': recognized and self.skip_alarm
            }
            
        except Exception as e:
            return {'recognized': False, 'error': str(e)}
