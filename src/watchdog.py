#!/home/gh/python/venv_py311/bin/python3
import cv2
from ultralytics import YOLO
import torch
import time
import os
import sys
import warnings
import pickle
from collections import deque
from db_logger import WatchdogDB
from video_writer import BrowserVideoWriter
from ir_detector import is_ir_mode, get_adaptive_confidence

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class SilentOutput:
    """Redirect stderr to suppress unwanted warnings"""
    def write(self, x): 
        pass
    def flush(self): 
        pass

sys.stdout = open('watchdog_output.log', 'a', buffering=1)
sys.stderr = SilentOutput()

# ─── FACE RECOGNITION HANDLER ─────────────────────────────────────
try:
    from face_handler import get_handler
    face_handler = get_handler('config.yaml')
    FACE_ENABLED = face_handler.enabled
except Exception as e:
    face_handler = None
    FACE_ENABLED = False
    print(f"⚠️ Face handler disabled: {e}")

# ─── KONFIGURATION ────────────────────────────────────────────────
RTSP_URL = "rtsp://gh2:Auchgut11@192.168.178.168:554/h264Preview_01_main"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;10000000"

THRESHOLD_Y = 628
THRESHOLD_X = 3000  # Vertical threshold (right quarter)
ALARM_RIGHT_QUARTER = True
RECORD_ALL_PERSONS = True  # Record when person detected, not just line crossing

BASE_RECORD_DIR = "/var/www/aufnahmen"
PRE_RECORD_SECONDS = 5
POST_RECORD_SECONDS = 30  # Aufnahme läuft 30s nach letzter Aktivität
FPS = 15
CHECK_INTERVAL = 0.5
TARGET_CLASSES = [0, 2, 3, 5, 7]

# ─── IR AUTO-DETECTION ────────────────────────────────────────────
# Will be set dynamically based on first frame
CONFIDENCE_TRACKING = 0.3
CONFIDENCE_DETECTION = 0.2
AI_RESOLUTION = 1536
MODE = "INITIALIZING"
IR_CHECK_INTERVAL = 30  # Re-check IR mode every 30 seconds
last_ir_check = 0

STATS_LOG = "watchdog_stats_live.txt"
STATS_PICKLE = "watchdog_counters.pkl"
STATS_LOG_MAX_SIZE = 100 * 1024  # 100 KB

# Classification alle 5 Minuten
CLASSIFICATION_INTERVAL = 300

# Face check alle 2 Sekunden pro Person
FACE_CHECK_INTERVAL = 2.0

# ─── MODELS ───────────────────────────────────────────────────────
model_detect = YOLO('yolov8m.pt').to('cuda')
model_cls = YOLO('yolov8m-cls.pt').to('cuda')

try:
    db = WatchdogDB()
    db_enabled = True
except Exception as e:
    db_enabled = False
    with open(STATS_LOG, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ DB disabled: {e}\n")

def log(msg):
    """Write log message with automatic rotation"""
    # Check if rotation needed
    if os.path.exists(STATS_LOG):
        log_size = os.path.getsize(STATS_LOG)
        if log_size > STATS_LOG_MAX_SIZE:
            # Rotate: keep last 50% of file
            try:
                with open(STATS_LOG, 'r') as f:
                    lines = f.readlines()
                
                # Keep last 50% of lines
                keep_lines = len(lines) // 2
                
                # Save to rotated file
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                rotated_file = f"watchdog_stats_{timestamp}.txt"
                with open(rotated_file, 'w') as f:
                    f.writelines(lines[:len(lines) - keep_lines])
                
                # Keep only recent lines in main log
                with open(STATS_LOG, 'w') as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] 🔄 Log rotated to {rotated_file}\n")
                    f.writelines(lines[-keep_lines:])
                
            except Exception as e:
                # If rotation fails, just truncate
                with open(STATS_LOG, 'w') as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ Log rotation failed: {e}\n")
    
    # Write log message
    with open(STATS_LOG, 'a') as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def load_counters():
    if os.path.exists(STATS_PICKLE):
        with open(STATS_PICKLE, 'rb') as f:
            return pickle.load(f)
    return {
        'start_time': time.time(),
        'alarm_count': 0,
        'total_frames': 0,
        'reconnect_count': 0,
        'face_recognized': 0,
        'face_unknown': 0
    }

def save_counters(c):
    with open(STATS_PICKLE, 'wb') as f:
        pickle.dump(c, f)

def get_monthly_dir():
    year = time.strftime("%Y")
    month = time.strftime("%m")
    path = os.path.join(BASE_RECORD_DIR, year, month)
    os.makedirs(path, exist_ok=True)
    return path

def reconnect():
    log("[RECONNECT]")
    time.sleep(2)
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

# ─── STARTUP ──────────────────────────────────────────────────────
counters = load_counters()
cap = reconnect()

ret, test_frame = cap.read()
if ret:
    FRAME_HEIGHT, FRAME_WIDTH = test_frame.shape[:2]
    log(f"📷 {FRAME_WIDTH}x{FRAME_HEIGHT} | DB:{'✅' if db_enabled else '❌'} | Face:{'✅' if FACE_ENABLED else '❌'}")
else:
    FRAME_WIDTH, FRAME_HEIGHT = 2560, 1440
    log(f"⚠️ Fallback: {FRAME_WIDTH}x{FRAME_HEIGHT}")

VERTICAL_THRESHOLD_X = THRESHOLD_X

out = None
recording_until = 0
recording_start = 0
current_video_id = None
current_video_path = None
alarm_trigger_type = None
alarm_trigger_id = None

track_history = {}
pre_buffer = deque(maxlen=PRE_RECORD_SECONDS * FPS)
last_check = 0
last_save = time.time()
last_scene = 0
last_classification = 0
error_count = 0
current_scene_stat_id = None

# Face check cooldown
face_check_cooldown = {}

log(f"🚀 Watchdog (auto-detect) | Vertikal @ x={VERTICAL_THRESHOLD_X} | Classification alle {CLASSIFICATION_INTERVAL}s")

# ─── MAIN LOOP ────────────────────────────────────────────────────
try:
    while True:
        now = time.time()
        
        if now - last_check < CHECK_INTERVAL:
            time.sleep(0.05)
            continue
        
        ret, frame = cap.read()
        if not ret:
            error_count += 1
            if error_count > 10:
                cap = reconnect()
                counters['reconnect_count'] += 1
                error_count = 0
            continue
        
        error_count = 0
        counters['total_frames'] += 1
        last_check = now
        pre_buffer.append(frame.copy())

        # ── IR MODE AUTO-DETECTION ────────────────────────────────────
        if now - last_ir_check > IR_CHECK_INTERVAL:
            config = get_adaptive_confidence(frame)
            CONFIDENCE_TRACKING = config['tracking']
            CONFIDENCE_DETECTION = config['detection']
            AI_RESOLUTION = config['resolution']
            MODE = config['mode']
            last_ir_check = now
            
            # Log mode changes
            if 'current_mode' not in globals():
                global current_mode
                current_mode = MODE
                log(f"🔍 Auto-detected: {MODE} mode (IR score: {config['ir_score']:.0f})")
            elif current_mode != MODE:
                log(f"🔄 Mode changed: {current_mode} → {MODE} (IR score: {config['ir_score']:.0f})")
                current_mode = MODE


        # ── 1. Scene Snapshot (alle 5s) ──────────────────────────────
        if now - last_scene > 5:
            results_detect = model_detect.predict(
                frame, device=0, classes=TARGET_CLASSES,
                verbose=False, imgsz=AI_RESOLUTION, conf=CONFIDENCE_DETECTION
            )
            
            counts = {'person': 0, 'car': 0, 'truck': 0, 'motorcycle': 0, 'bus': 0}
            for box in results_detect[0].boxes:
                label = model_detect.names[int(box.cls[0])]
                if label in counts:
                    counts[label] += 1
            
            if db_enabled:
                try:
                    current_scene_stat_id = db.log_scene_status(counts)
                except Exception as e:
                    log(f"[DB SCENE ERROR] {e}")
            
            last_scene = now

            if sum(counts.values()) > 0:
                log(f"📊 Scene: PKW:{counts['car']} LKW:{counts['truck']} Pers:{counts['person']}")
                
                # BACKUP: Keep recording if person in scene (even without tracking ID)
            if counts['person'] > 0:
                recording_until = max(recording_until, now + POST_RECORD_SECONDS)


        # ── 2. Image Classification (alle 5min) ───────────────────────
        if now - last_classification > CLASSIFICATION_INTERVAL:
            start_cls = time.time()
            results_cls = model_cls.predict(frame, imgsz=224, verbose=False)
            cls_time = (time.time() - start_cls) * 1000
            
            probs = results_cls[0].probs
            top5_ids = probs.top5
            top5_confs = probs.top5conf.tolist()
            
            top1_name = model_cls.names[top5_ids[0]]
            
            if db_enabled:
                try:
                    db.log_classification(
                        model_cls.names, top5_ids, top5_confs, cls_time,
                        video_id=current_video_id, scene_stat_id=current_scene_stat_id
                    )
                    log(f"🎯 Classification: {top1_name} ({top5_confs[0]:.2f}) | {cls_time:.0f}ms | DB:✅")
                except Exception as e:
                    log(f"🎯 Classification: {top1_name} ({top5_confs[0]:.2f}) | {cls_time:.0f}ms | DB:❌ {e}")
            else:
                log(f"🎯 Classification: {top1_name} ({top5_confs[0]:.2f}) | {cls_time:.0f}ms")
            
            last_classification = now

        # ── 3. Tracking + Face Recognition ────────────────────────────
        results_track = model_detect.track(
            frame, persist=True, device=0, classes=TARGET_CLASSES, 
            verbose=False, imgsz=AI_RESOLUTION, conf=CONFIDENCE_TRACKING
        )
        
        alarm_this_frame = False
        
        if results_track[0].boxes.id is not None:
            boxes = results_track[0].boxes.xyxy.cpu().numpy()
            ids   = results_track[0].boxes.id.cpu().numpy().astype(int)
            clss  = results_track[0].boxes.cls.cpu().numpy().astype(int)
            confs = results_track[0].boxes.conf.cpu().numpy()
            
            for box, id, cls, conf in zip(boxes, ids, clss, confs):
                x_center = (box[0] + box[2]) / 2
                y_center = (box[1] + box[3]) / 2
                label = model_detect.names[cls]
                in_upper = y_center < THRESHOLD_Y
                
                # ─── PERSON DETECTION RECORDING ───────────────────
                if label == "person" and RECORD_ALL_PERSONS:
                    alarm_this_frame = True
                    recording_until = now + POST_RECORD_SECONDS
                    if out is None:
                        alarm_trigger_type = "person_detected"
                        alarm_trigger_id = int(id)
                
                
                # ─── FACE RECOGNITION ─────────────────────────────────
                if label == "person" and in_upper and FACE_ENABLED:
                    if id not in face_check_cooldown or (now - face_check_cooldown[id]) > FACE_CHECK_INTERVAL:
                        face_check_cooldown[id] = now
                        
                        detection_data = {
                            'object_type': label,
                            'bbox': box.tolist(),
                            'in_upper_zone': in_upper,
                            'yolo_id': int(id),
                            'confidence': float(conf)
                        }
                        
                        try:
                            face_result = face_handler.process(frame, detection_data)
                            
                            if face_result:
                                if face_result.get('recognized'):
                                    counters['face_recognized'] += 1
                                    log(f"👤 ERKANNT: {face_result['name']} (d={face_result['distance']:.3f}) ID:{id}")
                                    
                                    # Skip alarm if configured
                                    if face_result.get('skip_alarm'):
                                        continue
                                else:
                                    # Check if it's a clustered unknown face
                                    cluster_id = face_result.get('cluster_id')
                                    if cluster_id:
                                        is_new = face_result.get('is_new_cluster', False)
                                        marker = "🆕" if is_new else "👥"
                                        counters['face_unknown'] += 1
                                        log(f"{marker} {cluster_id}: d={face_result.get('distance', 999):.3f} ID:{id}")
                                    else:
                                        counters['face_unknown'] += 1
                                        if face_result.get('reason') != 'no_face':
                                            log(f"👤 UNBEKANNT: d={face_result.get('distance', 999):.3f} ID:{id}")
                                
                                # DB Logging
                                if db_enabled:
                                    try:
                                        db.log_face_recognition(
                                            result=face_result,
                                            detection_id=None,
                                            video_id=current_video_id
                                        )
                                    except Exception as e:
                                        log(f"[DB FACE ERROR] {e}")
                        
                        except Exception as e:
                            log(f"[FACE ERROR] {e}")
                
                # ─── DB-LOGGING ───────────────────────────────────────
                if db_enabled:
                    try:
                        db.log_detection(
                            yolo_id=int(id),
                            object_type=label,
                            bbox=box.tolist(),
                            confidence=float(conf),
                            in_upper_zone=in_upper,
                            crossed_line=False,
                            frame_shape=frame.shape
                        )
                    except Exception as e:
                        log(f"[DB DETECTION ERROR] {e}")
                
                # ─── Alarm-Logik ──────────────────────────────────────
                if id in track_history:
                    old_x, old_y = track_history[id]
                    
                    # Horizontal
                    incoming_h = (old_y < THRESHOLD_Y and y_center >= THRESHOLD_Y)
                    outgoing_h = (old_y >= THRESHOLD_Y and y_center < THRESHOLD_Y)
                    
                    if incoming_h or outgoing_h:
                        direction = "IN" if incoming_h else "OUT"
                        counters['alarm_count'] += 1
                        log(f"🔔 #{counters['alarm_count']}: {label} {direction} H (ID:{id})")
                        alarm_this_frame = True
                        recording_until = now + POST_RECORD_SECONDS
                        if out is None:
                            recording_start = now
                            alarm_trigger_type = f"{label}_H_{direction}"
                            alarm_trigger_id = int(id)
                        save_counters(counters)
                        
                        # Mark crossed_line in DB
                        if db_enabled:
                            try:
                                db.log_detection(
                                    yolo_id=int(id), object_type=label,
                                    bbox=box.tolist(), confidence=float(conf),
                                    in_upper_zone=False, crossed_line=True,
                                    frame_shape=frame.shape
                                )
                            except Exception as e:
                                log(f"[DB CROSSED ERROR] {e}")
                    
                    # Vertikal
                    if ALARM_RIGHT_QUARTER:
                        incoming_v = (old_x < VERTICAL_THRESHOLD_X and x_center >= VERTICAL_THRESHOLD_X)
                        outgoing_v = (old_x >= VERTICAL_THRESHOLD_X and x_center < VERTICAL_THRESHOLD_X)
                        
                        if incoming_v or outgoing_v:
                            direction_v = "→RIGHT" if incoming_v else "←LEFT"
                            counters['alarm_count'] += 1
                            log(f"🔔 #{counters['alarm_count']}: {label} {direction_v} V (ID:{id})")
                            alarm_this_frame = True
                            recording_until = now + POST_RECORD_SECONDS
                            if out is None:
                                recording_start = now
                                alarm_trigger_type = f"{label}_V_{direction_v.replace(' ','')}"
                                alarm_trigger_id = int(id)
                            save_counters(counters)
                
                track_history[id] = (x_center, y_center)

        # ── 4. Video Recording ────────────────────────────────────────
        if alarm_this_frame or now < recording_until:
            if out is None:
                d = get_monthly_dir()
                filename = f"Alarm_{time.strftime('%d_%H-%M-%S')}.mp4"
                filepath = os.path.join(d, filename)
                current_video_path = filepath
                out = BrowserVideoWriter(filepath, FRAME_WIDTH, FRAME_HEIGHT, FPS)
                out.start()
                recording_start = now
                
                if db_enabled:
                    try:
                        current_video_id = db.start_recording(
                            filename=filename,
                            filepath=os.path.relpath(filepath, BASE_RECORD_DIR),
                            trigger_type=alarm_trigger_type or 'motion',
                            trigger_id=alarm_trigger_id or 0,
                            frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT, fps=FPS
                        )
                    except Exception as e:
                        log(f"⚠️ DB Start: {e}")
                
                log(f"💾 Video: {filename}")
                for bf in list(pre_buffer):
                    # Draw detection lines on frame
                    frame_with_overlay = bf.copy()
                    # Horizontal line (red)
                    cv2.line(frame_with_overlay, (0, THRESHOLD_Y), (FRAME_WIDTH, THRESHOLD_Y), (0, 0, 255), 3)
                    # Vertical line (blue)
                    cv2.line(frame_with_overlay, (VERTICAL_THRESHOLD_X, 0), (VERTICAL_THRESHOLD_X, FRAME_HEIGHT), (255, 0, 0), 3)
                    out.write(frame_with_overlay)
            
            if out:
                # Draw detection lines and bounding boxes
                frame_with_overlay = frame.copy()
                
                # Draw threshold lines
                cv2.line(frame_with_overlay, (0, THRESHOLD_Y), (FRAME_WIDTH, THRESHOLD_Y), (0, 0, 255), 3)
                cv2.line(frame_with_overlay, (VERTICAL_THRESHOLD_X, 0), (VERTICAL_THRESHOLD_X, FRAME_HEIGHT), (255, 0, 0), 3)
                
                # Draw bounding boxes and labels if tracking is active
                if results_track[0].boxes.id is not None:
                    for box, id, cls, conf in zip(boxes, ids, clss, confs):
                        # BBox
                        x1, y1, x2, y2 = map(int, box)
                        label = model_detect.names[cls]
                        color = (0, 255, 0) if label == "person" else (0, 255, 255)  # Green for person, yellow for vehicles
                        
                        cv2.rectangle(frame_with_overlay, (x1, y1), (x2, y2), color, 2)
                        
                        # Label with ID
                        text = f"{label} #{id} {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame_with_overlay, (x1, y1-text_height-10), (x1+text_width, y1), color, -1)
                        cv2.putText(frame_with_overlay, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                out.write(frame_with_overlay)
        else:
            if out:
                duration = now - recording_start + PRE_RECORD_SECONDS
                out.release()
                out = None
                
                if db_enabled and current_video_id:
                    try: 
                        db.finish_recording(current_video_id, current_video_path, duration)
                    except Exception as e:
                        log(f"⚠️ DB Finish: {e}")
                
                log(f"✅ Saved ({duration:.1f}s)")
                current_video_id = None

        # ── 5. Wartung ────────────────────────────────────────────────
        if now - last_save > 300:
            save_counters(counters)
            last_save = now
            if len(track_history) > 120:
                track_history.clear()
            # Cleanup face cooldown
            face_check_cooldown = {k: v for k, v in face_check_cooldown.items() 
                                  if k in ids or (now - v) < 60}

except KeyboardInterrupt:
    if out: 
        out.release()
    save_counters(counters)
    if db_enabled:
        db.close()
    log("🛑 Stop")
    if FACE_ENABLED:
        log(f"📊 Faces: recognized={counters.get('face_recognized', 0)} unknown={counters.get('face_unknown', 0)}")
    cap.release()
