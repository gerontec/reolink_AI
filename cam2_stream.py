#!/home/gh/python/venv_py311/bin/python3
"""
Cam2 Stream Monitor + Recorder + Analyzer (rebuilt)

Pipeline:
  1. YOLO monitors sub stream (640×360) continuously – no storage
  2. Person detected → ffmpeg records main stream (2560×1440) natively (-c copy) for 20 s
  3. After recording done: YOLO + InsightFace on saved raw clip
     → draw person / face bboxes per frame
     → pipe to ffmpeg libx264 → Firefox-compatible H.264 MP4
  4. Best frame saved as annotated JPG
  5. Results stored in DB (cam2_recordings, cam2_detected_*)
  6. run_cluster.sh + run_report.sh called for clustering and statistics
"""

import cv2
import os
import signal
import subprocess
import sys
import threading
import time
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ── Config ─────────────────────────────────────────────────────────────────────
SUB_RTSP        = "rtsp://admin:2einfach@192.168.178.128:554/h264Preview_01_sub"
MAIN_RTSP       = "rtsp://admin:2einfach@192.168.178.128:554/h264Preview_01_main"
YOLO_MODEL_PATH = "/opt/models/yolov8m.pt"
KNOWN_FACES_DIR = Path("/opt/known_faces")
OUTPUT_BASE     = Path("/var/www/web2")
CLUSTER_SCRIPT  = Path("/home/gh/python/reolink_AI/run_cluster.sh")
REPORT_SCRIPT   = Path("/home/gh/python/reolink_AI/run_report.sh")

DB_CONFIG = {
    "host": "localhost", "database": "wagodb",
    "user": "gh", "password": "a12345",
}

CONF_THRESHOLD  = 0.45   # YOLO person detection confidence
FACE_THRESHOLD  = 0.40   # InsightFace cosine-similarity match threshold
FRAME_SKIP      = 3      # sub-stream: analyse every N-th frame
ANALYSIS_SAMPLE = 10     # recorded video: run AI on every N-th frame
CLIP_DURATION   = 20     # seconds of main stream to record
CLIP_COOLDOWN   = 60     # minimum seconds between recordings
CLIP_MIN_SIZE   = 50_000 # bytes – discard clips smaller than this
RECONNECT_DELAY = 5      # seconds before RTSP reconnect
FFMPEG_TIMEOUT  = 8      # seconds for graceful ffmpeg shutdown

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cam2] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/gh/python/reolink_AI/logs/cam2_stream.log"),
    ],
)
log = logging.getLogger(__name__)

# ── Runtime state ────────────────────────────────────────────────────────────────
running        = True
last_clip_time = 0.0
clip_lock      = threading.Lock()
active_ffmpegs: List = []
ffmpeg_lock    = threading.Lock()
infer_lock     = threading.Lock()   # serialises GPU inference across threads

# ── AI models (set by _load_models) ─────────────────────────────────────────────
_device:      str = "cpu"
_yolo:        Optional[YOLO] = None
_face_app:    Optional[FaceAnalysis] = None
_known_enc:   List[np.ndarray] = []
_known_names: List[str] = []


# ══════════════════════════════════════════════════════════════════════════════════
# Signal handling
# ══════════════════════════════════════════════════════════════════════════════════

def _sigterm(sig, frame):
    global running
    log.info("Signal %s – stopping", sig)
    running = False
    _stop_all_ffmpegs()

signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT,  _sigterm)


# ══════════════════════════════════════════════════════════════════════════════════
# ffmpeg helpers
# ══════════════════════════════════════════════════════════════════════════════════

def _graceful_stop_ffmpeg(proc, label="ffmpeg"):
    if proc.poll() is not None:
        return
    try:
        proc.stdin.write(b"q\n")
        proc.stdin.flush()
        proc.wait(timeout=FFMPEG_TIMEOUT)
        log.debug("%s stopped cleanly", label)
    except subprocess.TimeoutExpired:
        log.warning("%s unresponsive – SIGTERM", label)
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            log.warning("%s – SIGKILL", label)
            proc.kill()
            proc.wait()
    except Exception as e:
        log.error("%s stop error: %s", label, e)


def _stop_all_ffmpegs():
    with ffmpeg_lock:
        procs = list(active_ffmpegs)
    for p in procs:
        _graceful_stop_ffmpeg(p)


# ══════════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════════

def _load_models():
    global _device, _yolo, _face_app

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if _device == "cuda" else "CPU"
    log.info("Device: %s (%s)", _device, gpu_name)

    log.info("Loading YOLO: %s", YOLO_MODEL_PATH)
    _yolo = YOLO(YOLO_MODEL_PATH)
    _yolo.to(_device)
    _yolo(torch.zeros(1, 3, 320, 320).to(_device), verbose=False)   # GPU warmup
    log.info("YOLO ready")

    log.info("Loading InsightFace buffalo_l…")
    providers = (["CUDAExecutionProvider"] if _device == "cuda"
                 else ["CPUExecutionProvider"])
    _face_app = FaceAnalysis(name="buffalo_l", providers=providers)
    _face_app.prepare(
        ctx_id=0 if _device == "cuda" else -1,
        det_size=(640, 640), det_thresh=0.3,
    )
    _face_app.get(np.zeros((64, 64, 3), dtype=np.uint8))   # warmup
    log.info("InsightFace ready")

    _load_known_faces()


def _load_known_faces():
    global _known_enc, _known_names
    if not KNOWN_FACES_DIR.exists():
        log.warning("Known-faces dir missing: %s", KNOWN_FACES_DIR)
        return
    files = (list(KNOWN_FACES_DIR.glob("*.jpg"))
             + list(KNOWN_FACES_DIR.glob("*.png")))
    for img_path in files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces = _face_app.get(img)
        if not faces:
            log.warning("No face in %s", img_path.name)
            continue
        emb = faces[0].embedding
        n   = np.linalg.norm(emb)
        _known_enc.append(emb / n if n > 0 else emb)
        _known_names.append(img_path.stem.replace("_", " "))
        log.info("  Known: %s", img_path.stem)
    log.info("%d known face(s) loaded", len(_known_names))


# ══════════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ══════════════════════════════════════════════════════════════════════════════════

def _yolo_persons(frame) -> List[Dict]:
    """YOLO person detection (sub-stream monitoring path). Uses infer_lock."""
    with infer_lock:
        res = _yolo(frame, verbose=False, conf=CONF_THRESHOLD, classes=[0])
    persons = []
    for box in res[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        persons.append({"bbox": (x1, y1, x2, y2), "conf": float(box.conf)})
    return persons


def _match_face(embedding: np.ndarray) -> Tuple[str, float]:
    if not _known_enc:
        return "Unknown", 0.0
    n   = np.linalg.norm(embedding)
    emb = embedding / n if n > 0 else embedding
    sims      = [float(np.dot(emb, k)) for k in _known_enc]
    best_idx  = int(np.argmax(sims))
    best_sim  = sims[best_idx]
    if best_sim > FACE_THRESHOLD:
        return _known_names[best_idx], best_sim
    return "Unknown", best_sim


def _detect_frame(frame) -> Dict[str, List]:
    """Full detection pass for recorded video analysis.
    YOLO persons first; InsightFace only when a person is present."""
    with infer_lock:
        yolo_res = _yolo(frame, verbose=False, conf=CONF_THRESHOLD, classes=[0])

    persons: List[Dict] = []
    person_bboxes: List[Tuple] = []
    for box in yolo_res[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        persons.append({"bbox": (x1, y1, x2, y2), "conf": float(box.conf)})
        person_bboxes.append((x1, y1, x2, y2))

    faces: List[Dict] = []
    if person_bboxes and _face_app is not None:
        with infer_lock:
            raw_faces = _face_app.get(frame)
        for face in raw_faces:
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            # discard faces not overlapping any person bbox
            if not any(
                fx1 < px2 and fx2 > px1 and fy1 < py2 and fy2 > py1
                for px1, py1, px2, py2 in person_bboxes
            ):
                continue
            name, conf = _match_face(face.embedding)
            det_score  = float(getattr(face, "det_score", 0.0))
            faces.append({
                "name":      name,
                "conf":      conf,
                "det_score": det_score,
                "bbox":      (fx1, fy1, fx2, fy2),
                "embedding": face.embedding,
            })

    return {"persons": persons, "faces": faces}


# ══════════════════════════════════════════════════════════════════════════════════
# Drawing
# ══════════════════════════════════════════════════════════════════════════════════

def _draw(frame: np.ndarray, dets: Dict) -> None:
    for p in dets.get("persons", []):
        x1, y1, x2, y2 = p["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {p['conf']:.2f}",
                    (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    for f in dets.get("faces", []):
        x1, y1, x2, y2 = f["bbox"]
        color = (0, 255, 255) if f["name"] != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{f['name']} {f['conf']:.2f}"
        cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ══════════════════════════════════════════════════════════════════════════════════
# Phase 2: analysis + annotation
# ══════════════════════════════════════════════════════════════════════════════════

def analyze_and_annotate(
    raw_path: Path,
    out_path: Path,
) -> Optional[Dict[str, Any]]:
    """
    Reads the natively recorded raw_path frame by frame.
    Every ANALYSIS_SAMPLE-th frame: YOLO + InsightFace.
    All frames: draw current detections.
    Pipes BGR frames to ffmpeg → libx264 H.264 MP4 (Firefox-compatible).
    Returns aggregated detections dict, or None on failure.
    """
    cap = cv2.VideoCapture(str(raw_path))
    if not cap.isOpened():
        log.error("Cannot open raw clip: %s", raw_path)
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Analysing %s  %dx%d @ %.1f fps", raw_path.name, w, h, fps)

    # ffmpeg encoder: rawvideo BGR → libx264 → faststart MP4
    enc_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y", str(out_path),
    ]
    enc = subprocess.Popen(
        enc_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_idx        = 0
    cur_dets: Dict   = {"persons": [], "faces": []}
    all_faces:   List[Dict] = []
    all_persons: List[Dict] = []
    best_score  = 0.0
    best_dets:  Optional[Dict]      = None
    best_frame: Optional[np.ndarray] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % ANALYSIS_SAMPLE == 0:
                cur_dets = _detect_frame(frame)
                all_persons.extend(cur_dets["persons"])
                all_faces.extend(cur_dets["faces"])

                # Track best frame: largest face × det_score wins
                score = 0.0
                for f in cur_dets["faces"]:
                    x1, y1, x2, y2 = f["bbox"]
                    area  = (x2 - x1) * (y2 - y1)
                    score = max(score, area * f["det_score"])
                if score > best_score:
                    best_score = score
                    best_dets  = {
                        "persons": list(cur_dets["persons"]),
                        "faces":   list(cur_dets["faces"]),
                    }
                    best_frame = frame.copy()

            _draw(frame, cur_dets)

            try:
                enc.stdin.write(frame.tobytes())
            except BrokenPipeError:
                log.error("Encoder pipe broken after frame %d", frame_idx)
                break

    finally:
        cap.release()
        try:
            enc.stdin.close()
        except Exception:
            pass

    try:
        enc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        log.warning("Encoder timeout – killing ffmpeg")
        enc.kill()
        enc.wait()

    if not (out_path.exists() and out_path.stat().st_size > 1_000):
        log.error("Annotated output missing or empty: %s", out_path)
        return None

    os.chmod(str(out_path), 0o664)
    log.info(
        "Annotated video: %s  %.1f MB  (%d frames, %d face detections)",
        out_path.name, out_path.stat().st_size / 1_048_576,
        frame_idx, len(all_faces),
    )
    return {
        "faces":       all_faces,
        "persons":     all_persons,
        "best_dets":   best_dets,
        "best_frame":  best_frame,
    }


def _save_best_frame(frame: np.ndarray, dets: Dict, out_path: Path) -> bool:
    img = frame.copy()
    _draw(img, dets)
    ok = cv2.imwrite(str(out_path), img)
    if ok:
        os.chmod(str(out_path), 0o664)
    return ok


# ══════════════════════════════════════════════════════════════════════════════════
# DB storage
# ══════════════════════════════════════════════════════════════════════════════════

def _save_to_db(
    out_path: Path,
    ts: datetime,
    analysis: Dict,
    annotated_img_rel: Optional[str] = None,
) -> Optional[int]:
    """Insert recording + detections. Returns recording_id or None."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cur  = conn.cursor()

        rel_path  = str(out_path.relative_to(OUTPUT_BASE))
        file_size = out_path.stat().st_size

        cur.execute("""
            INSERT INTO cam2_recordings
              (camera_name, file_path, file_type, file_size, recorded_at,
               analyzed, annotated_image_path, created_at)
            VALUES ('Camera2', %s, 'mp4', %s, %s, 1, %s, NOW())
            ON DUPLICATE KEY UPDATE
              file_size=VALUES(file_size),
              analyzed=1,
              annotated_image_path=VALUES(annotated_image_path)
        """, (rel_path, file_size, ts, annotated_img_rel))

        rec_id = cur.lastrowid
        if not rec_id:
            cur.execute("SELECT id FROM cam2_recordings WHERE file_path=%s", (rel_path,))
            row = cur.fetchone()
            rec_id = row[0] if row else None

        if rec_id:
            for tbl in ("cam2_detected_faces",
                        "cam2_detected_objects",
                        "cam2_analysis_summary"):
                cur.execute(f"DELETE FROM {tbl} WHERE recording_id=%s", (rec_id,))

            for f in analysis.get("faces", []):
                b = f["bbox"]
                emb_bytes = (f["embedding"].tobytes()
                             if f.get("embedding") is not None else None)
                cur.execute("""
                    INSERT INTO cam2_detected_faces
                      (recording_id, person_name, confidence,
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2, face_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (rec_id, f["name"], f["conf"],
                      b[0], b[1], b[2], b[3], emb_bytes))

            for p in analysis.get("persons", []):
                b = p["bbox"]
                cur.execute("""
                    INSERT INTO cam2_detected_objects
                      (recording_id, object_class, confidence,
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES (%s, 'person', %s, %s, %s, %s, %s)
                """, (rec_id, p["conf"], b[0], b[1], b[2], b[3]))

            faces_list   = analysis.get("faces", [])
            persons_list = analysis.get("persons", [])
            cur.execute("""
                INSERT INTO cam2_analysis_summary
                  (recording_id, total_faces, total_objects, total_vehicles,
                   max_persons, gpu_used, analyzed_at)
                VALUES (%s, %s, %s, 0, %s, %s, NOW())
            """, (rec_id,
                  len(faces_list),
                  len(persons_list),
                  len(persons_list),
                  _device == "cuda"))

        conn.commit()
        cur.close()
        conn.close()
        log.info("DB: recording_id=%s saved", rec_id)
        return rec_id

    except Exception as e:
        log.error("DB error: %s", e)
        return None


# ══════════════════════════════════════════════════════════════════════════════════
# Phase 1: native recording  →  Phase 2: analyse + annotate  →  Phase 3: DB + scripts
# ══════════════════════════════════════════════════════════════════════════════════

def record_and_analyze():
    """Runs in a daemon thread: record → analyse → annotate → DB → scripts."""
    ts      = datetime.now()
    ym      = ts.strftime("%Y/%m")
    stem    = f"Camera2_00_{ts.strftime('%Y%m%d_%H%M%S')}"
    out_dir = OUTPUT_BASE / ym
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{stem}_raw.mp4"
    out_path = out_dir / f"{stem}.mp4"
    ann_dir  = out_dir / "annotated"
    ann_dir.mkdir(exist_ok=True)
    best_jpg = ann_dir / f"best_{stem}.jpg"

    # ── Phase 1: native recording ─────────────────────────────────────────────
    log.info("Recording %s (%d s, native H.265)", stem, CLIP_DURATION)
    rec_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-rtsp_transport", "tcp",
        "-i", MAIN_RTSP,
        "-t", str(CLIP_DURATION),
        "-c", "copy",
        "-y", str(raw_path),
    ]
    proc = subprocess.Popen(
        rec_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    with ffmpeg_lock:
        active_ffmpegs.append(proc)

    try:
        proc.wait(timeout=CLIP_DURATION + 15)
    except subprocess.TimeoutExpired:
        log.warning("Recording timeout – stopping ffmpeg")
        _graceful_stop_ffmpeg(proc, stem)
    finally:
        with ffmpeg_lock:
            try:
                active_ffmpegs.remove(proc)
            except ValueError:
                pass

    if not raw_path.exists():
        log.warning("Raw clip missing: %s", raw_path.name)
        return

    raw_size = raw_path.stat().st_size
    if raw_size < CLIP_MIN_SIZE:
        log.warning("Raw clip too small (%d B) – discarding", raw_size)
        raw_path.unlink(missing_ok=True)
        return

    log.info("Raw clip saved: %s  %.1f MB", raw_path.name, raw_size / 1_048_576)

    # ── Phase 2: YOLO + InsightFace analysis + H.264 annotation ──────────────
    analysis = analyze_and_annotate(raw_path, out_path)

    if analysis is None or not out_path.exists():
        log.error("Annotation failed for %s", stem)
        return

    # ── Best frame JPG ────────────────────────────────────────────────────────
    ann_img_rel: Optional[str] = None
    if analysis.get("best_frame") is not None and analysis.get("best_dets"):
        if _save_best_frame(analysis["best_frame"], analysis["best_dets"], best_jpg):
            ann_img_rel = str(best_jpg.relative_to(OUTPUT_BASE))
            log.info("Best frame JPG: %s", best_jpg.name)

    # ── Phase 3: DB storage ───────────────────────────────────────────────────
    rec_id = _save_to_db(out_path, ts, analysis, ann_img_rel)

    # ── Phase 4: face clustering + report ────────────────────────────────────
    for script in (CLUSTER_SCRIPT, REPORT_SCRIPT):
        if script.exists():
            subprocess.Popen(
                [str(script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    log.info("Pipeline complete: %s  rec_id=%s", out_path.name, rec_id)


# ══════════════════════════════════════════════════════════════════════════════════
# Trigger
# ══════════════════════════════════════════════════════════════════════════════════

def trigger_clip():
    global last_clip_time
    now = time.time()
    with clip_lock:
        remaining = CLIP_COOLDOWN - (now - last_clip_time)
        if remaining > 0:
            log.debug("Cooldown %.0f s remaining", remaining)
            return
        last_clip_time = now
    threading.Thread(target=record_and_analyze, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════════
# Raw-file cleanup (keep for 3 days, then delete)
# ══════════════════════════════════════════════════════════════════════════════════

RAW_KEEP_DAYS = 3

def _cleanup_old_raws():
    """Delete *_raw.mp4 files older than RAW_KEEP_DAYS days."""
    cutoff = time.time() - RAW_KEEP_DAYS * 86_400
    count  = 0
    for raw in OUTPUT_BASE.rglob("*_raw.mp4"):
        try:
            if raw.stat().st_mtime < cutoff:
                raw.unlink()
                log.info("Cleanup: deleted %s", raw)
                count += 1
        except Exception as e:
            log.warning("Cleanup error for %s: %s", raw, e)
    if count:
        log.info("Cleanup: removed %d raw file(s) older than %d days", count, RAW_KEEP_DAYS)


# ══════════════════════════════════════════════════════════════════════════════════
# Main monitoring loop
# ══════════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=== cam2_stream start ===")
    _load_models()
    _cleanup_old_raws()

    log.info("Sub : %s", SUB_RTSP)
    log.info("Main: %s", MAIN_RTSP)
    log.info("Out : %s  clip=%ds  cooldown=%ds", OUTPUT_BASE, CLIP_DURATION, CLIP_COOLDOWN)

    frame_idx = 0

    while running:
        cap = cv2.VideoCapture(SUB_RTSP, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,  5_000)

        if not cap.isOpened():
            log.warning("Sub-stream unavailable – retry in %ds", RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)
            continue

        log.info("Sub-stream open")

        while running:
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame read error – reconnecting")
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            persons = _yolo_persons(frame)
            if not persons:
                continue

            max_conf = max(p["conf"] for p in persons)
            log.info("Person(s): %d  max_conf=%.2f", len(persons), max_conf)
            trigger_clip()

        cap.release()
        if running:
            log.info("Reconnecting in %ds…", RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)

    _stop_all_ffmpegs()
    log.info("=== cam2_stream stop ===")


if __name__ == "__main__":
    main()
