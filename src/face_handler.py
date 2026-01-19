#!/home/gh/python/venv_py311/bin/python3
"""
Face Recognition Handler - Separate module
Callable from person.py for face recognition tasks
"""
import sys
import os
import pickle
import numpy as np
from pathlib import Path

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class FaceHandler:
    def __init__(self, config_path='config.yaml'):
        self.enabled = False
        self.module_manager = None
        self.unknown_faces = {}  # {cluster_id: [embeddings]}
        self.unknown_faces_file = 'unknown_faces.pkl'
        self.cluster_threshold = 0.7  # Distance threshold for clustering
        self.next_cluster_id = 1
        
        # Load existing unknown faces
        self._load_unknown_faces()
        
        try:
            from modules import ModuleManager
            self.module_manager = ModuleManager(config_path)
            self.enabled = 'face_recognition' in self.module_manager.modules
            
            if self.enabled:
                print(f"✅ Face Recognition loaded ({len(self.unknown_faces)} unknown clusters)")
            else:
                print("⚠️ Face Recognition not enabled in config")
                
        except Exception as e:
            print(f"⚠️ Face Recognition init failed: {e}")
    
    def _load_unknown_faces(self):
        """Load previously seen unknown faces"""
        if os.path.exists(self.unknown_faces_file):
            try:
                with open(self.unknown_faces_file, 'rb') as f:
                    data = pickle.load(f)
                    self.unknown_faces = data.get('clusters', {})
                    self.next_cluster_id = data.get('next_id', 1)
            except Exception as e:
                print(f"⚠️ Could not load unknown faces: {e}")
    
    def _save_unknown_faces(self):
        """Save unknown face clusters"""
        try:
            data = {
                'clusters': self.unknown_faces,
                'next_id': self.next_cluster_id
            }
            with open(self.unknown_faces_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Could not save unknown faces: {e}")
    
    def _find_cluster(self, embedding):
        """
        Find matching cluster for embedding
        Returns: (cluster_id, distance) or (None, None) if no match
        """
        import torch
        
        if not self.unknown_faces:
            return None, None
        
        best_cluster = None
        best_distance = float('inf')
        
        for cluster_id, cluster_embeddings in self.unknown_faces.items():
            for known_emb in cluster_embeddings:
                if isinstance(known_emb, torch.Tensor):
                    dist = torch.dist(embedding, known_emb).item()
                else:
                    # Convert numpy to torch if needed
                    known_emb_tensor = torch.tensor(known_emb)
                    dist = torch.dist(embedding, known_emb_tensor).item()
                
                if dist < best_distance:
                    best_distance = dist
                    best_cluster = cluster_id
        
        if best_distance < self.cluster_threshold:
            return best_cluster, best_distance
        
        return None, None
    
    def _add_to_cluster(self, cluster_id, embedding):
        """Add embedding to existing cluster"""
        if cluster_id not in self.unknown_faces:
            self.unknown_faces[cluster_id] = []
        
        # Keep max 5 embeddings per cluster to avoid memory bloat
        if len(self.unknown_faces[cluster_id]) < 5:
            self.unknown_faces[cluster_id].append(embedding.cpu())
            self._save_unknown_faces()
    
    def _create_new_cluster(self, embedding):
        """Create new cluster for unknown face"""
        cluster_id = f"Unknown_{self.next_cluster_id}"
        self.unknown_faces[cluster_id] = [embedding.cpu()]
        self.next_cluster_id += 1
        self._save_unknown_faces()
        return cluster_id
    
    def process(self, frame, detection_data):
        """
        Process frame for face recognition
        
        Args:
            frame: numpy array - Full frame
            detection_data: dict with keys:
                - object_type: str
                - bbox: [x1, y1, x2, y2]
                - in_upper_zone: bool
                - yolo_id: int
                - confidence: float
        
        Returns:
            dict or None:
                - recognized: bool
                - name: str (person name or "Unknown_X")
                - distance: float
                - skip_alarm: bool (optional)
                - cluster_id: str (for unknown faces)
                - is_new_cluster: bool
        """
        if not self.enabled or not self.module_manager:
            return None
        
        # Only process persons in upper zone
        if detection_data.get('object_type') != 'person':
            return None
        
        if not detection_data.get('in_upper_zone', False):
            return None
        
        try:
            results = self.module_manager.process_detection(frame, detection_data)
            face_result = results.get('face_recognition') if results else None
            
            if not face_result:
                return None
            
            # If recognized - return as is
            if face_result.get('recognized'):
                return face_result
            
            # For unknown faces - try to cluster
            if face_result.get('reason') == 'no_face':
                return face_result
            
            # Get embedding from face recognition module
            face_module = self.module_manager.modules.get('face_recognition')
            if not face_module:
                return face_result
            
            # Extract embedding manually (we need to re-process to get it)
            try:
                import cv2
                import torch
                
                x1, y1, x2, y2 = map(int, detection_data['bbox'])
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    return face_result
                
                crop = frame[y1:y2, x1:x2]
                
                if crop.shape[0] < 50 or crop.shape[1] < 50:
                    return face_result
                
                # Get face embedding
                boxes, probs = face_module.detector.detect(crop)
                if boxes is None:
                    return face_result
                
                face_img = face_module.detector(crop)
                if face_img is None:
                    return face_result
                
                with torch.no_grad():
                    embedding = face_module.recognizer(face_img.unsqueeze(0).to('cuda'))
                
                # Try to find matching cluster
                cluster_id, cluster_dist = self._find_cluster(embedding)
                
                if cluster_id:
                    # Found existing cluster
                    self._add_to_cluster(cluster_id, embedding)
                    face_result['name'] = cluster_id
                    face_result['cluster_id'] = cluster_id
                    face_result['distance'] = cluster_dist
                    face_result['is_new_cluster'] = False
                else:
                    # Create new cluster
                    cluster_id = self._create_new_cluster(embedding)
                    face_result['name'] = cluster_id
                    face_result['cluster_id'] = cluster_id
                    face_result['is_new_cluster'] = True
                    face_result['distance'] = 0.0
                
            except Exception as e:
                print(f"Clustering error: {e}")
                return face_result
            
            return face_result
            
        except Exception as e:
            print(f"Face recognition error: {e}")
            return None

# Singleton instance
_face_handler = None

def get_handler(config_path='config.yaml'):
    """Get or create FaceHandler singleton"""
    global _face_handler
    if _face_handler is None:
        _face_handler = FaceHandler(config_path)
    return _face_handler

def process_face(frame, detection_data):
    """Convenience function for direct calls"""
    handler = get_handler()
    return handler.process(frame, detection_data)

def list_clusters():
    """List all unknown face clusters"""
    handler = get_handler()
    return {
        cluster_id: len(embeddings) 
        for cluster_id, embeddings in handler.unknown_faces.items()
    }

def merge_clusters(cluster_id_1, cluster_id_2):
    """Merge two clusters (if they're the same person)"""
    handler = get_handler()
    if cluster_id_1 in handler.unknown_faces and cluster_id_2 in handler.unknown_faces:
        handler.unknown_faces[cluster_id_1].extend(handler.unknown_faces[cluster_id_2])
        del handler.unknown_faces[cluster_id_2]
        handler._save_unknown_faces()
        return True
    return False

def rename_cluster(cluster_id, new_name):
    """Rename a cluster (when you identify who it is)"""
    handler = get_handler()
    if cluster_id in handler.unknown_faces:
        handler.unknown_faces[new_name] = handler.unknown_faces[cluster_id]
        del handler.unknown_faces[cluster_id]
        handler._save_unknown_faces()
        return True
    return False

# Test mode
if __name__ == "__main__":
    import cv2
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'list':
            # List clusters
            clusters = list_clusters()
            print(f"Unknown face clusters: {len(clusters)}")
            for cluster_id, count in clusters.items():
                print(f"  {cluster_id}: {count} embeddings")
        
        elif sys.argv[1] == 'merge' and len(sys.argv) >= 4:
            # Merge clusters
            cluster1, cluster2 = sys.argv[2], sys.argv[3]
            if merge_clusters(cluster1, cluster2):
                print(f"✅ Merged {cluster2} into {cluster1}")
            else:
                print(f"❌ Could not merge clusters")
        
        elif sys.argv[1] == 'rename' and len(sys.argv) >= 4:
            # Rename cluster
            cluster_id, new_name = sys.argv[2], sys.argv[3]
            if rename_cluster(cluster_id, new_name):
                print(f"✅ Renamed {cluster_id} to {new_name}")
            else:
                print(f"❌ Could not rename cluster")
        
        elif os.path.exists(sys.argv[1]):
            # Test with image
            handler = FaceHandler()
            print(f"Enabled: {handler.enabled}")
            
            frame = cv2.imread(sys.argv[1])
            detection_data = {
                'object_type': 'person',
                'bbox': [0, 0, frame.shape[1], frame.shape[0]],
                'in_upper_zone': True,
                'yolo_id': 1,
                'confidence': 0.9
            }
            
            result = handler.process(frame, detection_data)
            print(f"Result: {result}")
    else:
        handler = FaceHandler()
        print(f"Enabled: {handler.enabled}")
        print(f"Unknown clusters: {len(handler.unknown_faces)}")
