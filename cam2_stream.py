#!/home/gh/python/venv_py311/bin/python3
"""
Cam2 Stream Monitor + Main-Clip Recorder

Sub-Stream 640×360: kontinuierliche Personenerkennung via YOLO (kein Speichern)
Person erkannt → Main-Stream 2560×1440: 20s Clip aufzeichnen → run_chain.sh
"""

import cv2
import os
import signal
import subprocess
import sys
import threading
import time
import logging
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────────
SUB_RTSP     = "rtsp://admin:2einfach@192.168.178.128:554/h264Preview_01_sub"
MAIN_RTSP    = "rtsp://admin:2einfach@192.168.178.128:554/h264Preview_01_main"
YOLO_MODEL   = "/opt/models/yolov8m.pt"
OUTPUT_BASE  = Path("/var/www/web2")
CHAIN_SCRIPT = Path("/home/gh/python/reolink_AI/run_chain.sh")

CONF_THRESHOLD  = 0.45
FRAME_SKIP      = 3     # jeden N-ten Sub-Stream-Frame analysieren
CLIP_DURATION   = 25    # Sekunden Main-Stream aufnehmen
CLIP_COOLDOWN   = 90    # Sekunden Pause zwischen Clips (gleiche Person)
CLIP_MIN_SIZE   = 50_000  # Bytes — kleinere Clips werden verworfen
RECONNECT_DELAY = 5     # Sekunden vor RTSP-Reconnect

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cam2] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/home/gh/python/reolink_AI/logs/cam2_stream.log"),
    ],
)
log = logging.getLogger(__name__)

# ── State ──────────────────────────────────────────────────────────────────────
running        = True
last_clip_time = 0.0
clip_lock      = threading.Lock()


def _sigterm(sig, frame):
    global running
    log.info("Signal %s — stopping", sig)
    running = False


signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT,  _sigterm)


# ── Clip recorder ──────────────────────────────────────────────────────────────

def record_main_clip():
    """Zeichnet CLIP_DURATION Sekunden vom Main-Stream auf (läuft in Thread)."""
    ts        = datetime.now()
    ym        = ts.strftime("%Y/%m")
    filename  = f"Camera2_00_{ts.strftime('%Y%m%d_%H%M%S')}.mp4"
    out_dir   = OUTPUT_BASE / ym
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / filename

    log.info("📹 Clip starten: %s (%ds)", filename, CLIP_DURATION)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-rtsp_transport", "tcp",
        "-i", MAIN_RTSP,
        "-t", str(CLIP_DURATION),
        "-c", "copy",
        "-y", str(out_path),
    ]

    try:
        subprocess.run(cmd, timeout=CLIP_DURATION + 15)
    except subprocess.TimeoutExpired:
        log.warning("⏰ Clip-Timeout (%ds)", CLIP_DURATION + 15)
    except Exception as e:
        log.error("Clip-Fehler: %s", e)
        return

    if not out_path.exists():
        log.warning("⚠ Clip-Datei fehlt: %s", filename)
        return

    size = out_path.stat().st_size
    if size < CLIP_MIN_SIZE:
        log.warning("⚠ Clip zu klein (%.0f B) — verworfen", size)
        out_path.unlink(missing_ok=True)
        return

    os.chmod(str(out_path), 0o664)
    log.info("✅ Clip: %s  %.1f MB", filename, size / 1024 / 1024)

    # Chain triggern
    if CHAIN_SCRIPT.exists():
        base = str(OUTPUT_BASE / ym)
        subprocess.Popen(
            [str(CHAIN_SCRIPT), "--base-path", base, "--limit", "5"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        log.info("🔗 Chain: %s", base)


def trigger_clip():
    """Startet Clip-Thread wenn Cooldown abgelaufen."""
    global last_clip_time
    now = time.time()
    with clip_lock:
        remaining = CLIP_COOLDOWN - (now - last_clip_time)
        if remaining > 0:
            log.debug("Cooldown (%.0fs verbleibend)", remaining)
            return
        last_clip_time = now
    log.info("🎬 Clip-Aufnahme gestartet")
    threading.Thread(target=record_main_clip, daemon=True).start()


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    log.info("=== cam2_stream start ===")
    log.info("Sub : %s", SUB_RTSP)
    log.info("Main: %s", MAIN_RTSP)
    log.info("Out : %s  Clip=%ds  Cooldown=%ds", OUTPUT_BASE, CLIP_DURATION, CLIP_COOLDOWN)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s (%s)", device,
             torch.cuda.get_device_name(0) if device == "cuda" else "CPU")

    model = YOLO(YOLO_MODEL)
    model.to(device)
    model(torch.zeros(1, 3, 320, 320).to(device), verbose=False)
    log.info("YOLO ready")

    frame_idx = 0

    while running:
        cap = cv2.VideoCapture(SUB_RTSP, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,  5_000)

        if not cap.isOpened():
            log.warning("Sub-Stream nicht erreichbar — Retry in %ds", RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)
            continue

        log.info("Sub-Stream offen")

        while running:
            ret, frame = cap.read()
            if not ret:
                log.warning("Frame-Lesefehler — Reconnect")
                break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            # YOLO: nur Person-Klasse (schnell, kein Overhead für Fahrzeuge etc.)
            results = model(frame, verbose=False, conf=CONF_THRESHOLD, classes=[0])
            n = len(results[0].boxes)
            if n == 0:
                continue

            confs = [float(b.conf) for b in results[0].boxes]
            log.info("👤 %d Person(en) — max conf %.2f", n, max(confs))
            trigger_clip()

        cap.release()
        if running:
            log.info("Reconnect in %ds …", RECONNECT_DELAY)
            time.sleep(RECONNECT_DELAY)

    log.info("=== cam2_stream stop ===")


if __name__ == "__main__":
    main()
