#!/home/gh/python/venv_py311/bin/python3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import hashlib
import os
import json

DB_HOST = "192.168.178.218"
DB_USER = "gh"
DB_PASS = "a12345"
DB_NAME = "wagodb"

engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}', 
                       pool_pre_ping=True, pool_recycle=3600, echo=False)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class VideoArchive(Base):
    __tablename__ = 'cam_video_archive'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), unique=True, index=True)
    filepath = Column(String(512))
    recorded_at = Column(DateTime, default=datetime.now, index=True)
    duration_seconds = Column(Float)
    filesize_mb = Column(Float)
    trigger_object_type = Column(String(20))
    trigger_object_id = Column(Integer)
    frame_width = Column(Integer)
    frame_height = Column(Integer)
    fps = Column(Integer)
    total_detections = Column(Integer, default=0)
    archived = Column(Boolean, default=False)
    notes = Column(String(512), nullable=True)

class DetectedObject(Base):
    __tablename__ = 'cam_detected_objects'
    id = Column(Integer, primary_key=True, autoincrement=True)
    object_hash = Column(String(64), unique=True, index=True)
    object_type = Column(String(20), index=True)
    first_seen = Column(DateTime, default=datetime.now)
    last_seen = Column(DateTime, default=datetime.now)
    total_detections = Column(Integer, default=1)
    times_crossed_line = Column(Integer, default=0)
    first_video_id = Column(Integer, ForeignKey('cam_video_archive.id'), nullable=True)
    
class Detection(Base):
    __tablename__ = 'cam_detections'
    id = Column(Integer, primary_key=True, autoincrement=True)
    object_hash = Column(String(64), index=True)
    video_id = Column(Integer, ForeignKey('cam_video_archive.id'), nullable=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    object_type = Column(String(20))
    yolo_id = Column(Integer)
    confidence = Column(Float)
    bbox_x1 = Column(Float)
    bbox_y1 = Column(Float)
    bbox_x2 = Column(Float)
    bbox_y2 = Column(Float)
    in_upper_zone = Column(Boolean, default=False)
    crossed_line = Column(Boolean, default=False)

class SceneStats(Base):
    __tablename__ = 'cam_scene_stats'
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    person_count = Column(Integer, default=0)
    car_count = Column(Integer, default=0)
    truck_count = Column(Integer, default=0)
    motorcycle_count = Column(Integer, default=0)
    bus_count = Column(Integer, default=0)
    total_objects = Column(Integer, default=0)

class ImageClassification(Base):
    __tablename__ = 'cam_image_classification'
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, ForeignKey('cam_video_archive.id'), nullable=True)
    scene_stat_id = Column(Integer, ForeignKey('cam_scene_stats.id'), nullable=True)
    classified_at = Column(DateTime, default=datetime.now)
    model_name = Column(String(64), default='yolov8m-cls')
    img_size = Column(Integer, default=224)
    top1_class_id = Column(Integer)
    top1_class_name = Column(String(128))
    top1_confidence = Column(Float)
    top2_class_name = Column(String(128), nullable=True)
    top2_confidence = Column(Float, nullable=True)
    top3_class_name = Column(String(128), nullable=True)
    top3_confidence = Column(Float, nullable=True)
    top5_confidence_sum = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=True)

class FaceRecognition(Base):
    __tablename__ = 'cam_face_recognitions'
    id = Column(Integer, primary_key=True, autoincrement=True)
    detection_id = Column(Integer, ForeignKey('cam_detections.id'), nullable=True)
    video_id = Column(Integer, ForeignKey('cam_video_archive.id'), nullable=True)
    timestamp = Column(DateTime, default=datetime.now, index=True)
    recognized = Column(Boolean, default=False)
    person_name = Column(String(128), nullable=True, index=True)
    distance = Column(Float)
    face_confidence = Column(Float, nullable=True)
    face_bbox_x = Column(Integer, nullable=True)
    face_bbox_y = Column(Integer, nullable=True)
    face_bbox_w = Column(Integer, nullable=True)
    face_bbox_h = Column(Integer, nullable=True)

class FaceEmbedding(Base):
    __tablename__ = 'cam_face_embeddings'
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(128), index=True)  # Unknown_1, Unknown_2, enrolled name
    embedding_vector = Column(Text)  # JSON array of 512 floats
    face_quality = Column(Float)  # Quality score (0-1)
    bbox_width = Column(Integer)
    bbox_height = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    last_seen = Column(DateTime, default=datetime.now)
    match_count = Column(Integer, default=1)
    is_enrolled = Column(Boolean, default=False)
    source = Column(String(64), default='auto')  # 'auto' or 'manual'
    notes = Column(String(512), nullable=True)

class DailyStats(Base):
    __tablename__ = 'cam_daily_stats'
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, unique=True, index=True)
    total_detections = Column(Integer, default=0)
    person_count = Column(Integer, default=0)
    car_count = Column(Integer, default=0)
    motorcycle_count = Column(Integer, default=0)
    bus_count = Column(Integer, default=0)
    truck_count = Column(Integer, default=0)
    line_crossings = Column(Integer, default=0)
    upper_zone_movements = Column(Integer, default=0)
    videos_recorded = Column(Integer, default=0)
    total_video_duration = Column(Float, default=0)

Base.metadata.create_all(engine)

def generate_object_hash(bbox, object_type, frame_shape):
    x1, y1, x2, y2 = bbox
    height, width = frame_shape[:2]
    rel_x = (x1 + x2) / 2 / width
    rel_y = (y1 + y2) / 2 / height
    rel_w = (x2 - x1) / width
    rel_h = (y2 - y1) / height
    grid_x = round(rel_x * 10) / 10
    grid_y = round(rel_y * 10) / 10
    grid_w = round(rel_w * 10) / 10
    grid_h = round(rel_h * 10) / 10
    hash_str = f"{object_type}_{grid_x:.1f}_{grid_y:.1f}_{grid_w:.1f}_{grid_h:.1f}"
    return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

class WatchdogDB:
    def __init__(self):
        self.session = Session()
        self.current_video_id = None
        self.object_cache = {}
    
    def start_recording(self, filename, filepath, trigger_type, trigger_id, 
                       frame_width, frame_height, fps):
        try:
            video = VideoArchive(
                filename=filename, filepath=filepath,
                trigger_object_type=trigger_type, trigger_object_id=trigger_id,
                frame_width=frame_width, frame_height=frame_height, fps=fps
            )
            self.session.add(video)
            self.session.commit()
            self.current_video_id = video.id
            return video.id
        except:
            self.session.rollback()
            return None
    
    def finish_recording(self, video_id, filepath, duration_seconds=None):
        try:
            video = self.session.query(VideoArchive).filter_by(id=video_id).first()
            if video and os.path.exists(filepath):
                video.filesize_mb = os.path.getsize(filepath) / (1024 * 1024)
                if duration_seconds:
                    video.duration_seconds = duration_seconds
                else:
                    detections = self.session.query(Detection).filter_by(video_id=video_id).all()
                    if detections:
                        first = min(d.timestamp for d in detections)
                        last = max(d.timestamp for d in detections)
                        video.duration_seconds = (last - first).total_seconds()
                det_count = self.session.query(Detection).filter_by(video_id=video_id).count()
                video.total_detections = det_count
                self.session.commit()
                self.update_daily_stats(videos_recorded=1, total_duration=video.duration_seconds or 0)
                return True
            return False
        except:
            self.session.rollback()
            return False
    
    def log_detection(self, yolo_id, object_type, bbox, confidence, in_upper_zone, crossed_line, frame_shape):
        obj_hash = generate_object_hash(bbox, object_type, frame_shape)
        
        if obj_hash in self.object_cache:
            obj = self.object_cache[obj_hash]
            is_new = False
        else:
            obj = self.session.query(DetectedObject).filter_by(object_hash=obj_hash).first()
            if obj:
                self.object_cache[obj_hash] = obj
                is_new = False
            else:
                obj = DetectedObject(object_hash=obj_hash, object_type=object_type, first_video_id=self.current_video_id)
                self.session.add(obj)
                try:
                    self.session.flush()
                    self.object_cache[obj_hash] = obj
                    is_new = True
                except:
                    self.session.rollback()
                    obj = self.session.query(DetectedObject).filter_by(object_hash=obj_hash).first()
                    if not obj:
                        return None, False
                    self.object_cache[obj_hash] = obj
                    is_new = False
        
        obj.last_seen = datetime.now()
        obj.total_detections += 1
        if crossed_line:
            obj.times_crossed_line += 1
        
        detection = Detection(
            object_hash=obj_hash, video_id=self.current_video_id, object_type=object_type,
            yolo_id=yolo_id, confidence=confidence,
            bbox_x1=bbox[0], bbox_y1=bbox[1], bbox_x2=bbox[2], bbox_y2=bbox[3],
            in_upper_zone=in_upper_zone, crossed_line=crossed_line
        )
        self.session.add(detection)
        self.session.commit()
        return obj_hash, is_new
    
    def log_scene_status(self, counts):
        try:
            scene = SceneStats(
                person_count=counts.get('person', 0), car_count=counts.get('car', 0),
                truck_count=counts.get('truck', 0), motorcycle_count=counts.get('motorcycle', 0),
                bus_count=counts.get('bus', 0), total_objects=sum(counts.values())
            )
            self.session.add(scene)
            self.session.commit()
            return scene.id
        except:
            self.session.rollback()
            return None
    
    def log_classification(self, model_names, top5_ids, top5_confs, inference_ms, video_id=None, scene_stat_id=None):
        try:
            cls = ImageClassification(
                video_id=video_id, scene_stat_id=scene_stat_id,
                top1_class_id=int(top5_ids[0]), top1_class_name=model_names[top5_ids[0]], top1_confidence=float(top5_confs[0]),
                top2_class_name=model_names[top5_ids[1]] if len(top5_ids) > 1 else None,
                top2_confidence=float(top5_confs[1]) if len(top5_confs) > 1 else None,
                top3_class_name=model_names[top5_ids[2]] if len(top5_ids) > 2 else None,
                top3_confidence=float(top5_confs[2]) if len(top5_confs) > 2 else None,
                top5_confidence_sum=float(sum(top5_confs)), inference_time_ms=inference_ms
            )
            self.session.add(cls)
            self.session.commit()
            return True
        except:
            self.session.rollback()
            return False
    
    def log_face_recognition(self, result, detection_id=None, video_id=None):
        """Log face recognition result"""
        try:
            face_rec = FaceRecognition(
                detection_id=detection_id,
                video_id=video_id,
                recognized=result.get('recognized', False),
                person_name=result.get('name'),
                distance=result.get('distance', 999.0),
                face_confidence=result.get('face_confidence'),
                face_bbox_x=result.get('face_bbox', [None]*4)[0],
                face_bbox_y=result.get('face_bbox', [None]*4)[1],
                face_bbox_w=result.get('face_bbox', [None]*4)[2],
                face_bbox_h=result.get('face_bbox', [None]*4)[3]
            )
            self.session.add(face_rec)
            self.session.commit()
            return True
        except:
            self.session.rollback()
            return False
    
    def store_face_embedding(self, person_id, embedding, face_quality, bbox_size, is_enrolled=False, source='auto'):
        """Store high-quality face embedding in database"""
        try:
            # Convert tensor to list if needed
            if hasattr(embedding, 'cpu'):
                embedding = embedding.cpu().numpy().tolist()
            elif hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            # Flatten if nested
            if isinstance(embedding[0], list):
                embedding = embedding[0]
            
            # Store as JSON
            embedding_json = json.dumps(embedding)
            
            face_emb = FaceEmbedding(
                person_id=person_id,
                embedding_vector=embedding_json,
                face_quality=face_quality,
                bbox_width=bbox_size[0],
                bbox_height=bbox_size[1],
                is_enrolled=is_enrolled,
                source=source
            )
            self.session.add(face_emb)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            print(f"Store embedding error: {e}")
            return False
    
    def get_face_embeddings(self, person_id=None, min_quality=0.0):
        """Retrieve face embeddings from database"""
        try:
            query = self.session.query(FaceEmbedding)
            
            if person_id:
                query = query.filter_by(person_id=person_id)
            
            if min_quality > 0:
                query = query.filter(FaceEmbedding.face_quality >= min_quality)
            
            embeddings = query.all()
            
            result = []
            for emb in embeddings:
                result.append({
                    'person_id': emb.person_id,
                    'embedding': json.loads(emb.embedding_vector),
                    'quality': emb.face_quality,
                    'match_count': emb.match_count,
                    'created_at': emb.created_at
                })
            
            return result
        except:
            return []
    
    def update_embedding_match(self, person_id):
        """Increment match counter when embedding is used"""
        try:
            embeddings = self.session.query(FaceEmbedding).filter_by(person_id=person_id).all()
            for emb in embeddings:
                emb.last_seen = datetime.now()
                emb.match_count += 1
            self.session.commit()
        except:
            self.session.rollback()
    
    def update_daily_stats(self, videos_recorded=0, total_duration=0, crossed_line=False):
        try:
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            stats = self.session.query(DailyStats).filter_by(date=today).first()
            if not stats:
                stats = DailyStats(date=today)
                self.session.add(stats)
            if videos_recorded:
                stats.videos_recorded += videos_recorded
            if total_duration:
                stats.total_video_duration += total_duration
            if crossed_line:
                stats.line_crossings += 1
            self.session.commit()
        except:
            self.session.rollback()
    
    def close(self):
        try:
            self.session.commit()
        except:
            pass
        self.session.close()
