#!/home/gh/python/venv_py311/bin/python3
"""
Cam2 Real-time Stream Processor
Liest RTSP-Stream von cam2, erkennt Personen mit YOLO (Tesla P4),
trägt Events in cam2_* Tabellen ein analog zur cam1 Chain.
"""

import cv2
import time
import logging
import signal
import sys
import os
from datetime import datetime
from pathlib import Path

import mysql.connector
import torch
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
RTSP_URL      = "rtsp://admin:2einfach@192.168.178.128:554/Preview_01_sub"
YOLO_MODEL    = "/opt/models/yolov8m.pt"
SNAPSHOT_DIR  = Path("/var/www/web1/cam2_snapshots")
CAMERA_NAME   = "Camera2"

DETECT_CLASSES    = {"person", "car", "truck", "motorcycle", "bus"}
PERSON_CLASS      = "person"
CONF_THRESHOLD    = 0.45
FRAME_SKIP        = 2          # analyse every Nth frame
SESSION_COOLDOWN  = 30         # seconds without detection → close session
RECONNECT_DELAY   = 5          # seconds before RTSP reconnect

DB_CONFIG = {
    "host": "localhost",
    "user": "gh",
    "password": "a12345",
    "database": "wagodb",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cam2_stream] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/gh/python/reolink_AI/logs/cam2_stream.log"),
    ],
)
log = logging.getLogger(__name__)

# ── DB helpers ────────────────────────────────────────────────────────────────

def db_connect():
    return mysql.connector.connect(**DB_CONFIG)


def db_open_recording(conn, recorded_at: datetime) -> int:
    cur = conn.cursor()
    # synthetic path so unique constraint is happy
    path = f"stream/cam2/{recorded_at.strftime('%Y%m%d_%H%M%S')}"
    cur.execute(
        """INSERT INTO cam2_recordings
           (camera_name, file_path, file_type, file_size, recorded_at, analyzed, created_at)
           VALUES (%s, %s, 'jpg', 0, %s, 0, NOW())""",
        (CAMERA_NAME, path, recorded_at),
    )
    conn.commit()
    rid = cur.lastrowid
    cur.close()
    return rid


def db_insert_object(conn, recording_id: int, cls: str, conf: float, bbox):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO cam2_detected_objects
           (recording_id, object_class, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (recording_id, cls, float(conf), x1, y1, x2, y2),
    )
    conn.commit()
    cur.close()


def db_close_recording(conn, recording_id: int, total_obj: int, total_persons: int,
                        total_vehicles: int, max_persons: int, gpu: bool):
    cur = conn.cursor()
    cur.execute("UPDATE cam2_recordings SET analyzed=1 WHERE id=%s", (recording_id,))
    cur.execute(
        """INSERT INTO cam2_analysis_summary
           (recording_id, total_faces, total_objects, total_vehicles,
            max_persons, scene_category, gpu_used, analyzed_at)
           VALUES (%s, 0, %s, %s, %s, %s, %s, NOW())""",
        (recording_id, total_obj, total_vehicles, max_persons,
         "person" if total_persons > 0 else "vehicle" if total_vehicles > 0 else "motion",
         int(gpu)),
    )
    conn.commit()
    cur.close()


# ── Session state ─────────────────────────────────────────────────────────────

class Session:
    def __init__(self, recording_id: int, started: datetime):
        self.recording_id = recording_id
        self.started      = started
        self.last_detect  = time.time()
        self.total_obj    = 0
        self.total_persons = 0
        self.total_vehicles = 0
        self.max_persons  = 0
        self.snapshot_saved = False


# ── Main loop ─────────────────────────────────────────────────────────────────

running = True

def _sigterm(sig, frame):
    global running
    log.info("Signal %s — shutting down", sig)
    running = False

signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT,  _sigterm)


def main():
    log.info("=== cam2_stream starting ===")
    log.info("RTSP: %s", RTSP_URL)

    # GPU / model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s  (%s)", device,
             torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    model = YOLO(YOLO_MODEL)
    model.to(device)
    # warmup
    model(torch.zeros(1, 3, 320, 320).to(device), verbose=False)
    log.info("YOLO model loaded & warmed up")

    conn = db_connect()
    log.info("DB connected")

    session: Session | None = None
    frame_idx = 0

    while running:
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            log.warning("Cannot open RTSP stream — retry in %ds", RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)
            continue

        log.info("Stream open — processing")

        while running:
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame read failed — reconnecting")
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            now = time.time()

            # ── close stale session ──────────────────────────────────────────
            if session and (now - session.last_detect) > SESSION_COOLDOWN:
                db_close_recording(
                    conn, session.recording_id,
                    session.total_obj, session.total_persons,
                    session.total_vehicles, session.max_persons,
                    device == "cuda",
                )
                log.info("Session %d closed  persons=%d  objects=%d  duration=%.0fs",
                         session.recording_id, session.total_persons, session.total_obj,
                         now - time.mktime(session.started.timetuple()))
                session = None

            # ── inference ───────────────────────────────────────────────────
            results  = model(frame, verbose=False, conf=CONF_THRESHOLD)
            boxes    = results[0].boxes
            detected = []
            for box in boxes:
                cls_name = model.names[int(box.cls)]
                if cls_name in DETECT_CLASSES:
                    detected.append((cls_name, float(box.conf), box.xyxy[0].tolist()))

            if not detected:
                continue

            # ── open session if needed ───────────────────────────────────────
            if session is None:
                ts = datetime.now()
                rid = db_open_recording(conn, ts)
                session = Session(rid, ts)
                log.info("Session %d opened  %s", rid, ts.strftime("%H:%M:%S"))

            session.last_detect = now
            persons_this_frame = sum(1 for d in detected if d[0] == PERSON_CLASS)
            session.max_persons = max(session.max_persons, persons_this_frame)

            for cls_name, conf, bbox in detected:
                db_insert_object(conn, session.recording_id, cls_name, conf, bbox)
                session.total_obj += 1
                if cls_name == PERSON_CLASS:
                    session.total_persons += 1
                elif cls_name in {"car", "truck", "bus", "motorcycle"}:
                    session.total_vehicles += 1

            # ── snapshot (first frame per session with person) ───────────────
            if not session.snapshot_saved and persons_this_frame > 0:
                snap_name = f"cam2_{session.started.strftime('%Y%m%d_%H%M%S')}.jpg"
                snap_path = SNAPSHOT_DIR / snap_name
                cv2.imwrite(str(snap_path), frame)
                # store path in recording
                cur = conn.cursor()
                cur.execute("UPDATE cam2_recordings SET annotated_image_path=%s WHERE id=%s",
                            (f"cam2_snapshots/{snap_name}", session.recording_id))
                conn.commit()
                cur.close()
                session.snapshot_saved = True
                log.info("Snapshot → %s", snap_name)

            log.debug("Frame %d: %s", frame_idx,
                      ", ".join(f"{c}({v:.0%})" for c, v, _ in detected))

        cap.release()
        if running:
            time.sleep(RECONNECT_DELAY)

    # ── shutdown ──────────────────────────────────────────────────────────────
    if session:
        db_close_recording(
            conn, session.recording_id,
            session.total_obj, session.total_persons,
            session.total_vehicles, session.max_persons,
            device == "cuda",
        )
        log.info("Final session %d closed on shutdown", session.recording_id)

    conn.close()
    log.info("=== cam2_stream stopped ===")


if __name__ == "__main__":
    main()
