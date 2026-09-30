#!/home/gh/python/venv_py311/bin/python3
"""
Cam2 Stream Monitor + Recorder + Analyzer  –  OO-Version

Klassen-Übersicht:
  Config          – alle Konstanten an einem Ort
  Database        – MySQL-Zugriff (recordings, faces, objects, summary)
  ModelManager    – YOLO + InsightFace laden, Inferenz, Known-Faces
  Annotator       – Bounding-Boxes zeichnen + Best-Frame speichern
  ClipRecorder    – Phase 1 (native ffmpeg) + Phase 2 (annotate) + Phase 3 (DB/Scripts)
  StreamMonitor   – Haupt-Loop: Sub-Stream lesen, YOLO, Trigger
  Cam2App         – Einstiegspunkt: alles verdrahten, Signal-Handler
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


# ══════════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════════

class Config:
    SUB_RTSP        = "rtsp://admin:2einfach@192.168.178.128:554/h264Preview_01_sub"
    MAIN_RTSP       = "rtsp://admin:2einfach@192.168.178.128:554/h264Preview_01_main"
    YOLO_MODEL_PATH = "/opt/models/yolov8m.pt"
    KNOWN_FACES_DIR = Path("/opt/known_faces")
    OUTPUT_BASE     = Path("/var/www/web2")
    CLUSTER_SCRIPT  = Path("/home/gh/python/reolink_AI/run_cluster.sh")
    REPORT_SCRIPT   = Path("/home/gh/python/reolink_AI/run_report.sh")
    LOG_FILE        = "/home/gh/python/reolink_AI/logs/cam2_stream.log"

    DB = {
        "host": "localhost", "database": "wagodb",
        "user": "gh", "password": "a12345",
    }

    CONF_THRESHOLD  = 0.45   # YOLO person detection confidence
    FACE_THRESHOLD  = 0.45   # InsightFace cosine-similarity match threshold (0.45: 0 Fremd-Fehlmatch bei 80er-gh-Bank)
    TRACK_SIM       = 0.50   # cosine: Zugehoerigkeit zum selben Gesichts-Track (#1 Aggregation)
    SUB_FACE_THRESH = 0.50   # InsightFace det_score on sub-stream → Aufnahme-Trigger
    FRAME_SKIP      = 3      # sub-stream: analyse every N-th frame
    ANALYSIS_SAMPLE = 10     # recorded video: run AI on every N-th frame
    CLIP_DURATION   = 20     # seconds of main stream to record
    CLIP_COOLDOWN   = 29     # minimum seconds between recordings (29 < 30s Clip: leichte Ueberlappung moeglich)
    CLIP_MIN_SIZE   = 50_000 # bytes – discard clips smaller than this
    RECONNECT_DELAY = 5      # seconds before RTSP reconnect
    FFMPEG_TIMEOUT  = 8      # seconds for graceful ffmpeg shutdown
    RAW_KEEP_DAYS   = 3      # delete *_raw.mp4 files older than this

    # Output-Clip: an die gemessene WiFi-Physik angepasst.
    # cam2 auf f7240: RSSI ~45dB (~-68dBm), HT bis 216 Mbit; Main-Quelle nur ~1,1 Mbit/s.
    # → Link trägt native 1080p locker, kein Downscale nötig (volle Auflösung fürs
    #   Gesichts-Erkennen). Bitrate moderat gedeckelt, faststart bleibt.
    REC_HEIGHT      = 1080   # native Main-Höhe (scale=-2:1080 = no-op bei 1920x1080)
    REC_CRF         = 23     # bessere Qualität (Bandbreite ist reichlich da)
    REC_MAXRATE_K   = 3000   # kbit/s Spitzen-Bitrate-Deckel (Bewegungs-Headroom)
    REC_BUFSIZE_K   = 6000   # kbit VBV-Puffer


# ══════════════════════════════════════════════════════════════════════════════════
# Logging (Modul-Level, wird von allen Klassen genutzt)
# ══════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [cam2] %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        __import__('logging.handlers', fromlist=['x']).RotatingFileHandler('/tmp/cam2-ai.log', maxBytes=10*1024*1024, backupCount=1),
    ],
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════════
# Database
# ══════════════════════════════════════════════════════════════════════════════════

class Database:
    """Kapselt alle MySQL-Operationen für cam2."""

    def __init__(self, db_config: Dict):
        self._config = db_config

    # ── interne Helpers ────────────────────────────────────────────────────────

    def _connect(self):
        return mysql.connector.connect(**self._config)

    # ── öffentliche API ────────────────────────────────────────────────────────

    def save_recording(
        self,
        out_path: Path,
        ts: datetime,
        analysis: Dict,
        output_base: Path,
        annotated_img_rel: Optional[str] = None,
    ) -> Optional[int]:
        """Insert recording + detections. Gibt recording_id zurück oder None."""
        try:
            conn = self._connect()
            cur  = conn.cursor()

            rel_path  = str(out_path.relative_to(output_base))
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
                cur.execute(
                    "SELECT id FROM cam2_recordings WHERE file_path=%s", (rel_path,)
                )
                row = cur.fetchone()
                rec_id = row[0] if row else None

            if rec_id:
                self._delete_old_detections(cur, rec_id)
                self._insert_faces(cur, rec_id, analysis.get("faces", []))
                self._insert_persons(cur, rec_id, analysis.get("persons", []))
                self._insert_summary(cur, rec_id, analysis)

            conn.commit()
            cur.close()
            conn.close()
            log.info("DB: recording_id=%s gespeichert", rec_id)
            return rec_id

        except Exception as e:
            log.error("DB Fehler: %s", e)
            return None

    # ── private Insert-Helpers ─────────────────────────────────────────────────

    def _delete_old_detections(self, cur, rec_id: int):
        for tbl in ("cam2_detected_faces", "cam2_detected_objects",
                    "cam2_analysis_summary"):
            cur.execute(f"DELETE FROM {tbl} WHERE recording_id=%s", (rec_id,))

    def _insert_faces(self, cur, rec_id: int, faces: List[Dict]):
        for f in faces:
            b = f["bbox"]
            emb_bytes = (f["embedding"].tobytes()
                         if f.get("embedding") is not None else None)
            cur.execute("""
                INSERT INTO cam2_detected_faces
                  (recording_id, person_name, confidence, det_score,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2, face_embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (rec_id, f["name"], f["conf"], f.get("det_score", 0.0),
                  b[0], b[1], b[2], b[3], emb_bytes))

    def _insert_persons(self, cur, rec_id: int, persons: List[Dict]):
        for p in persons:
            b = p["bbox"]
            cur.execute("""
                INSERT INTO cam2_detected_objects
                  (recording_id, object_class, confidence,
                   bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                VALUES (%s, 'person', %s, %s, %s, %s, %s)
            """, (rec_id, p["conf"], b[0], b[1], b[2], b[3]))

    def _insert_summary(self, cur, rec_id: int, analysis: Dict):
        faces   = analysis.get("faces", [])
        persons = analysis.get("persons", [])
        gpu     = analysis.get("gpu_used", False)
        cur.execute("""
            INSERT INTO cam2_analysis_summary
              (recording_id, total_faces, total_objects, total_vehicles,
               max_persons, gpu_used, analyzed_at)
            VALUES (%s, %s, %s, 0, %s, %s, NOW())
        """, (rec_id, len(faces), len(persons), len(persons), gpu))


# ══════════════════════════════════════════════════════════════════════════════════
# ModelManager
# ══════════════════════════════════════════════════════════════════════════════════

class ModelManager:
    """Lädt YOLO + InsightFace, hält Known-Faces, kapselt alle Inferenz-Aufrufe."""

    def __init__(self, cfg: Config):
        self._cfg          = cfg
        self.device        = "cuda" if torch.cuda.is_available() else "cpu"
        self._yolo:        Optional[YOLO]         = None
        self._face_app:    Optional[FaceAnalysis] = None
        self._known_enc:   List[np.ndarray]       = []
        self._known_names: List[str]              = []
        self._lock         = threading.Lock()   # serialisiert GPU-Inferenz

    # ── Laden ──────────────────────────────────────────────────────────────────

    def load(self):
        gpu_name = (torch.cuda.get_device_name(0)
                    if self.device == "cuda" else "CPU")
        log.info("Device: %s (%s)", self.device, gpu_name)
        self._load_yolo()
        self._load_insightface()
        self._load_known_faces()

    def _load_yolo(self):
        log.info("Lade YOLO: %s", self._cfg.YOLO_MODEL_PATH)
        self._yolo = YOLO(self._cfg.YOLO_MODEL_PATH)
        self._yolo.to(self.device)
        self._yolo(
            torch.zeros(1, 3, 320, 320).to(self.device), verbose=False
        )  # Warmup
        log.info("YOLO bereit")

    def _load_insightface(self):
        log.info("Lade InsightFace buffalo_l…")
        providers = (["CUDAExecutionProvider"] if self.device == "cuda"
                     else ["CPUExecutionProvider"])
        self._face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._face_app.prepare(
            ctx_id=0 if self.device == "cuda" else -1,
            det_size=(640, 640), det_thresh=0.3,
        )
        self._face_app.get(np.zeros((64, 64, 3), dtype=np.uint8))  # Warmup
        log.info("InsightFace bereit")

    def _load_known_faces(self):
        d = self._cfg.KNOWN_FACES_DIR
        if not d.exists():
            log.warning("Known-Faces Verzeichnis fehlt: %s", d)
            return
        files = list(d.glob("*.jpg")) + list(d.glob("*.png"))
        for img_path in files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = self._face_app.get(img)
            if not faces:
                log.warning("Kein Gesicht in %s", img_path.name)
                continue
            emb = faces[0].embedding
            n   = np.linalg.norm(emb)
            self._known_enc.append(emb / n if n > 0 else emb)
            self._known_names.append(img_path.stem.replace("_", " "))
            log.info("  Known: %s", img_path.stem)

        # Embedding-Banken (.npy): Matrix (N,512) L2-normierter Embeddings, alle
        # unter dem Datei-Stem als Name -> robuste Multi-Referenz pro Person
        # (z.B. gh.npy = viele gh-Gesichter aus einem Clip). _match_face nimmt
        # die max-Similarity ueber die Bank.
        for npy_path in sorted(d.glob("*.npy")):
            try:
                arr = np.load(str(npy_path))
            except Exception as e:
                log.warning("Kann %s nicht laden: %s", npy_path.name, e)
                continue
            if arr.ndim == 1:
                arr = arr[None, :]
            bank_name = npy_path.stem.replace("_", " ")
            cnt = 0
            for emb in arr:
                nn = float(np.linalg.norm(emb))
                self._known_enc.append(emb / nn if nn > 0 else emb)
                self._known_names.append(bank_name)
                cnt += 1
            log.info("  Known-Bank: %s (%d Embeddings)", npy_path.stem, cnt)

        log.info("%d bekannte(s) Gesicht(er) geladen", len(self._known_names))

    # ── Inferenz ───────────────────────────────────────────────────────────────

    def detect_persons(self, frame) -> List[Dict]:
        """Schnelle Person-Erkennung für den Sub-Stream (ohne InsightFace)."""
        with self._lock:
            res = self._yolo(
                frame, verbose=False,
                conf=self._cfg.CONF_THRESHOLD, classes=[0]
            )
        persons = []
        for box in res[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            persons.append({"bbox": (x1, y1, x2, y2), "conf": float(box.conf)})
        return persons

    def detect_faces(self, frame) -> List:
        """InsightFace auf einem Frame – nur Gesichter über det_score-Schwelle.

        Wird im Sub-Stream als Aufnahme-Trigger genutzt: erst ein echtes
        Gesicht (nicht bloß eine Person) löst die Aufnahme aus.
        """
        if self._face_app is None:
            return []
        with self._lock:
            raw_faces = self._face_app.get(frame)
        return [f for f in raw_faces
                if float(getattr(f, "det_score", 0.0)) >= self._cfg.SUB_FACE_THRESH]

    def detect_full(self, frame) -> Dict[str, List]:
        """YOLO (Personen) + InsightFace (Gesichter) für aufgezeichnetes Video.

        Gesichter werden unabhängig von Personen-Erkennung gespeichert.
        """
        with self._lock:
            yolo_res = self._yolo(
                frame, verbose=False,
                conf=self._cfg.CONF_THRESHOLD, classes=[0]
            )

        persons: List[Dict] = []
        for box in yolo_res[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            persons.append({"bbox": (x1, y1, x2, y2), "conf": float(box.conf)})

        # Gesichter UNABHÄNGIG von YOLO-Personen erkennen & speichern:
        # Embedding-Matrix landet auch bei Nahaufnahmen ohne sichtbaren Körper
        # in der DB. Personen-Boxen dienen nur noch zur Annotierung.
        faces: List[Dict] = []
        if self._face_app is not None:
            with self._lock:
                raw_faces = self._face_app.get(frame)
            for face in raw_faces:
                det = float(getattr(face, "det_score", 0.0))
                if det < self._cfg.SUB_FACE_THRESH:
                    continue
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                name, conf = self._match_face(face.embedding)
                faces.append({
                    "name":      name,
                    "conf":      conf,
                    "det_score": det,
                    "bbox":      (fx1, fy1, fx2, fy2),
                    "embedding": face.embedding,
                })

        return {"persons": persons, "faces": faces}

    def _match_face(self, embedding: np.ndarray) -> Tuple[str, float]:
        if not self._known_enc:
            return "Unknown", 0.0
        n   = np.linalg.norm(embedding)
        emb = embedding / n if n > 0 else embedding
        sims     = [float(np.dot(emb, k)) for k in self._known_enc]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]
        if best_sim > self._cfg.FACE_THRESHOLD:
            return self._known_names[best_idx], best_sim
        return "Unknown", best_sim


# ══════════════════════════════════════════════════════════════════════════════════
# Annotator
# ══════════════════════════════════════════════════════════════════════════════════

class Annotator:
    """Zeichnet Bounding-Boxes auf Frames und speichert Best-Frame-JPGs."""

    @staticmethod
    def draw(frame: np.ndarray, dets: Dict) -> None:
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
            cv2.putText(frame, f"{f['name']} {f['conf']:.2f}",
                        (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    @staticmethod
    def save_best_frame(frame: np.ndarray, dets: Dict, out_path: Path) -> bool:
        img = frame.copy()
        Annotator.draw(img, dets)
        ok = cv2.imwrite(str(out_path), img)
        if ok:
            os.chmod(str(out_path), 0o664)
        return ok


# ══════════════════════════════════════════════════════════════════════════════════
# ClipRecorder
# ══════════════════════════════════════════════════════════════════════════════════

class ClipRecorder:
    """
    Verwaltet den gesamten Aufnahme-Pipeline:
      Phase 1 – Native ffmpeg Aufnahme (H.265 copy)
      Phase 2 – YOLO + InsightFace Analyse + H.264 Annotierung
      Phase 3 – DB-Speicherung + externe Skripte
    """

    def __init__(self, cfg: Config, models: ModelManager, db: Database):
        self._cfg    = cfg
        self._models = models
        self._db     = db
        self._annot  = Annotator()

        self._last_clip_time = 0.0
        self._clip_lock      = threading.Lock()
        self._active_ffmpegs: List = []
        self._ffmpeg_lock    = threading.Lock()

    # ── Trigger ────────────────────────────────────────────────────────────────

    def trigger(self):
        """Cooldown prüfen → wenn OK, Pipeline in Daemon-Thread starten."""
        now = time.time()
        with self._clip_lock:
            remaining = self._cfg.CLIP_COOLDOWN - (now - self._last_clip_time)
            if remaining > 0:
                log.debug("Cooldown %.0f s verbleibend", remaining)
                return
            self._last_clip_time = now
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def stop_all(self):
        with self._ffmpeg_lock:
            procs = list(self._active_ffmpegs)
        for p in procs:
            self._stop_ffmpeg(p)

    # ── Haupt-Pipeline ─────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_dir(d: Path):
        """Verzeichnis anlegen und auf 2775 (setgid, group-writable) setzen.

        Die übergeordneten /var/www/web2-Ordner sind Gruppe www-data + setgid,
        daher erben neue Unterordner Gruppe www-data. setgid+group-write sorgt
        dafür, dass der www-data-Cron (Retention 60 Tage) die Inhalte löschen darf.
        """
        d.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(str(d), 0o2775)
        except PermissionError:
            pass

    def _run_pipeline(self):
        ts      = datetime.now()
        ym      = ts.strftime("%Y/%m")
        stem    = f"Camera2_00_{ts.strftime('%Y%m%d_%H%M%S')}"
        out_dir = self._cfg.OUTPUT_BASE / ym
        self._ensure_dir(out_dir)

        raw_path = out_dir / f"{stem}_raw.mp4"
        out_path = out_dir / f"{stem}.mp4"
        ann_dir  = out_dir / "annotated"
        self._ensure_dir(ann_dir)
        best_jpg = ann_dir / f"best_{stem}.jpg"

        # Phase 1
        if not self._record_raw(raw_path, stem):
            return

        # Phase 2
        analysis = self._annotate(raw_path, out_path)
        if analysis is None or not out_path.exists():
            log.error("Annotierung fehlgeschlagen: %s", stem)
            return

        # Fehl-Trigger: Phase-2 fand weder Person noch Gesicht -> Clip verwerfen
        # (kein DB-Eintrag, Dateien loeschen).
        if analysis.get("n_persons_total", 0) == 0 and analysis.get("n_faces_total", 0) == 0:
            log.info("Phase-2: 0 Personen / 0 Gesichter - Clip verworfen: %s", stem)
            for _p in (out_path, raw_path):
                try:
                    _p.unlink()
                except FileNotFoundError:
                    pass
            return

        # Best-Frame JPG
        ann_img_rel: Optional[str] = None
        if analysis.get("best_frame") is not None and analysis.get("best_dets"):
            if self._annot.save_best_frame(
                analysis["best_frame"], analysis["best_dets"], best_jpg
            ):
                ann_img_rel = str(best_jpg.relative_to(self._cfg.OUTPUT_BASE))
                log.info("Best-Frame JPG: %s", best_jpg.name)

        # Phase 3
        analysis["gpu_used"] = self._models.device == "cuda"
        rec_id = self._db.save_recording(
            out_path, ts, analysis, self._cfg.OUTPUT_BASE, ann_img_rel
        )

        # Externe Skripte
        for script in (self._cfg.CLUSTER_SCRIPT, self._cfg.REPORT_SCRIPT):
            if script.exists():
                subprocess.Popen(
                    [str(script)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

        self._write_debug(raw_path, ts)
        log.info("Pipeline abgeschlossen: %s  rec_id=%s", out_path.name, rec_id)

    # ── Debug-Info (Raw-Clip) ──────────────────────────────────────────────────

    def _write_debug(self, raw_path: Path, ts: datetime):
        debug_path = raw_path.with_suffix(".debug.txt")
        import json as _json

        # ffprobe: Metadaten
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-print_format", "json",
                 "-show_streams", "-show_format", str(raw_path)],
                capture_output=True, text=True, timeout=15,
            )
            info = _json.loads(r.stdout) if r.stdout else {}
        except Exception as e:
            info = {"error": str(e)}

        fmt     = info.get("format", {})
        vs      = next((s for s in info.get("streams", [])
                        if s.get("codec_type") == "video"), {})
        aus     = next((s for s in info.get("streams", [])
                        if s.get("codec_type") == "audio"), None)

        size_b  = raw_path.stat().st_size if raw_path.exists() else 0
        dur     = float(fmt.get("duration", 0))
        br      = int(fmt.get("bit_rate", 0))
        codec   = vs.get("codec_name", "?")
        profile = vs.get("profile", "")
        pix_fmt = vs.get("pix_fmt", "?")
        w, h    = vs.get("width", "?"), vs.get("height", "?")
        nb_frames_int = int(vs.get("nb_frames", 0) or 0)
        fps_real = nb_frames_int / dur if dur and nb_frames_int else None
        fps_str  = f"{fps_real:.2f}" if fps_real else vs.get("r_frame_rate", "?")

        codec_str = f"{codec} ({profile})" if profile else codec
        if codec in ("h264", "avc1"):
            mime = "video/mp4; codecs=avc1"
        elif codec in ("hevc", "h265"):
            mime = "video/mp4; codecs=hev1  [!] Firefox unterstützt HEVC nicht nativ"
        else:
            mime = f"video/mp4 (codec={codec})"

        audio_str = (f"{aus['codec_name']} {aus.get('sample_rate','?')} Hz"
                     if aus else "kein Audio")


        # ffmpeg Fehler-Scan: dekodiert alle Frames, sammelt Fehlermeldungen
        try:
            scan = subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "error",
                 "-i", str(raw_path), "-f", "null", "-"],
                capture_output=True, text=True, timeout=self._cfg.CLIP_DURATION + 30,
            )
            error_lines = [l.strip() for l in scan.stderr.splitlines() if l.strip()]
        except Exception as e:
            error_lines = [f"Scan-Fehler: {e}"]

        lines = [
            f"=== Raw-Clip Debug: {raw_path.name} ===",
            f"Zeitstempel  : {ts.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "── Datei ────────────────────────────────────────",
            f"  Pfad         : {raw_path}",
            f"  Größe        : {size_b / 1_048_576:.2f} MB  ({size_b:,} Bytes)",
            f"  Dauer        : {dur:.1f} s",
            f"  Bitrate      : {br // 1000} kbps" if br else "  Bitrate      : ?",
            "",
            "── Video-Stream ─────────────────────────────────",
            f"  Codec        : {codec_str}",
            f"  Auflösung    : {w}x{h}",
            f"  FPS (real)   : {fps_str}  ({nb_frames_int} Frames / {dur:.1f} s)",
            f"  Pixelformat  : {pix_fmt}",
            f"  MIME         : {mime}",
            "",
            "── Audio-Stream ─────────────────────────────────",
            f"  {audio_str}",
            "",
            f"── Bildfehler-Scan ({len(error_lines)} Zeile(n)) ──────────────────",
        ]
        if error_lines:
            lines += [f"  {l}" for l in error_lines]
        else:
            lines.append("  keine Fehler gefunden")
        lines.append("")

        try:
            debug_path.write_text("\n".join(lines), encoding="utf-8")
            os.chmod(str(debug_path), 0o664)
            log.info("Raw-Debug: %s  (%d Fehlerzeile(n))",
                     debug_path.name, len(error_lines))
        except Exception as e:
            log.warning("Debug-Datei Fehler: %s", e)

    # ── Phase 1: Native Aufnahme ───────────────────────────────────────────────

    def _record_raw(self, raw_path: Path, stem: str) -> bool:
        log.info("Aufnahme %s (%d s, native H.265)", stem, self._cfg.CLIP_DURATION)
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-i", self._cfg.MAIN_RTSP,
            "-t", str(self._cfg.CLIP_DURATION),
            "-c", "copy",
            "-y", str(raw_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with self._ffmpeg_lock:
            self._active_ffmpegs.append(proc)

        try:
            proc.wait(timeout=self._cfg.CLIP_DURATION + 15)
        except subprocess.TimeoutExpired:
            log.warning("Aufnahme Timeout – stoppe ffmpeg")
            self._stop_ffmpeg(proc, stem)
        finally:
            with self._ffmpeg_lock:
                try:
                    self._active_ffmpegs.remove(proc)
                except ValueError:
                    pass

        if not raw_path.exists():
            log.warning("Raw-Clip fehlt: %s", raw_path.name)
            return False

        raw_size = raw_path.stat().st_size
        if raw_size < self._cfg.CLIP_MIN_SIZE:
            log.warning("Raw-Clip zu klein (%d B) – verwerfen", raw_size)
            raw_path.unlink(missing_ok=True)
            return False

        os.chmod(str(raw_path), 0o664)
        log.info("Raw-Clip gespeichert: %s  %.1f MB",
                 raw_path.name, raw_size / 1_048_576)
        return True

    # ── Phase 2: Analyse + Annotierung ────────────────────────────────────────

    @staticmethod
    def _probe_avg_fps(path: Path) -> Optional[float]:
        """Echte Durchschnitts-fps (avg_frame_rate) via ffprobe; None bei Fehler."""
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=avg_frame_rate",
                 "-of", "default=nk=1:nw=1", str(path)],
                capture_output=True, text=True, timeout=15,
            )
            num, _, den = r.stdout.strip().partition("/")
            d = float(den) if den else 1.0
            val = float(num) / d if d else 0.0
            return val if val > 0 else None
        except Exception:
            return None

    def _track_aggregate_match(self, best_faces, all_faces):
        """#1 Track-Aggregation: pro Best-Frame-Gesicht alle Embeddings derselben
        Person ueber den Clip mitteln und EINMAL matchen (stabiler als Einzelframe).
        Ueberschreibt name/conf der Best-Frame-Gesichter mit der Track-Entscheidung;
        Bbox/Crop/gespeichertes Embedding bleiben unveraendert."""
        if not best_faces or not all_faces:
            return
        def _n(v):
            m = np.linalg.norm(v)
            return v / m if m > 0 else v
        pool = [(_n(g["embedding"]), g["embedding"]) for g in all_faces]
        for f in best_faces:
            ref = _n(f["embedding"])
            embs = [raw for nn, raw in pool if float(np.dot(ref, nn)) >= self._cfg.TRACK_SIM]
            if not embs:
                embs = [f["embedding"]]
            mean = np.mean(embs, axis=0)
            old_name, old_conf = f.get("name"), float(f.get("conf", 0.0))
            name, conf = self._models._match_face(mean)
            f["name"], f["conf"], f["track_size"] = name, conf, len(embs)
            log.info("  Track-Aggregat: %s conf=%.3f aus %d Frames (Einzelframe war %s/%.3f)",
                     name, conf, len(embs), old_name, old_conf)

    def _annotate(
        self, raw_path: Path, out_path: Path
    ) -> Optional[Dict[str, Any]]:
        cap = cv2.VideoCapture(str(raw_path))
        if not cap.isOpened():
            log.error("Kann Raw-Clip nicht öffnen: %s", raw_path)
            return None

        # Reolink meldet nominales r_frame_rate ~2x der echten Rate; cv2 liest
        # diesen Nominalwert -> Re-Encode mit -r erzeugte sonst "Zeitraffer".
        # Daher echte Durchschnitts-fps (avg_frame_rate) aus dem Clip ziehen.
        fps = self._probe_avg_fps(raw_path) or cap.get(cv2.CAP_PROP_FPS) or 25.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Analysiere %s  %dx%d @ %.1f fps", raw_path.name, w, h, fps)

        # Output: mittlere Auflösung (Höhe REC_HEIGHT, Breite -2 = AR erhalten),
        # WiFi-tauglicher Bitraten-Deckel, H.264 High/yuv420p + faststart →
        # Firefox spielt die Datei per Klick sofort ab (moov-Atom am Anfang).
        enc_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "pipe:0",
            "-vf", f"scale=-2:{self._cfg.REC_HEIGHT}",
            "-vcodec", "libx264", "-preset", "fast",
            "-profile:v", "high", "-level", "4.0",
            "-crf", str(self._cfg.REC_CRF),
            "-maxrate", f"{self._cfg.REC_MAXRATE_K}k",
            "-bufsize", f"{self._cfg.REC_BUFSIZE_K}k",
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
        best_dets:  Optional[Dict]       = None
        best_frame: Optional[np.ndarray] = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx % self._cfg.ANALYSIS_SAMPLE == 0:
                    cur_dets = self._models.detect_full(frame)
                    all_persons.extend(cur_dets["persons"])
                    all_faces.extend(cur_dets["faces"])

                    # Frame-Score: Gesichter dominieren, Personen als Fallback.
                    # Nur der EINE beste Frame wird gespeichert; ausschließlich
                    # seine Detektionen landen in der DB → Frontend-Crops stimmen
                    # immer (kein Cropping aus fremden Frames = kein "Rasen").
                    fscore = sum(
                        (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
                        * f["det_score"] for f in cur_dets["faces"]
                    )
                    pscore = sum(
                        (p["bbox"][2] - p["bbox"][0]) * (p["bbox"][3] - p["bbox"][1])
                        * p["conf"] for p in cur_dets["persons"]
                    )
                    frame_score = fscore * 1_000_000 + pscore
                    if frame_score > best_score and (cur_dets["faces"] or cur_dets["persons"]):
                        best_score = frame_score
                        best_dets  = {
                            "persons": list(cur_dets["persons"]),
                            "faces":   list(cur_dets["faces"]),
                        }
                        best_frame = frame.copy()

                self._annot.draw(frame, cur_dets)

                try:
                    enc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    log.error("Encoder Pipe unterbrochen nach Frame %d", frame_idx)
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
            log.warning("Encoder Timeout – ffmpeg wird beendet")
            enc.kill()
            enc.wait()

        if not (out_path.exists() and out_path.stat().st_size > 1_000):
            log.error("Annotiertes Video fehlt oder leer: %s", out_path)
            return None

        os.chmod(str(out_path), 0o664)
        # Nur die Detektionen des Best-Frames speichern (passen zum gespeicherten
        # best_*.jpg → Frontend-Crop korrekt). all_faces/all_persons dienten nur
        # der Frame-Auswahl/Live-Annotation.
        best_dets = best_dets or {"persons": [], "faces": []}

        # #1 Track-Level-Aggregation: Identitaet NICHT pro Einzelframe entscheiden.
        # Pro Best-Frame-Gesicht alle clip-weiten Embeddings derselben Person
        # (Cosine >= TRACK_SIM) mitteln und EINMAL gegen die Galerie matchen ->
        # rauschstabile Zuordnung. Bbox/Crop bleiben vom Best-Frame.
        self._track_aggregate_match(best_dets["faces"], all_faces)

        log.info(
            "Annotiertes Video: %s  %.1f MB  (%d Frames; Best-Frame: %d Gesicht(er) / %d Person(en); %d Face-Detektionen gesamt)",
            out_path.name, out_path.stat().st_size / 1_048_576,
            frame_idx, len(best_dets["faces"]), len(best_dets["persons"]), len(all_faces),
        )
        return {
            "faces":      best_dets["faces"],
            "persons":    best_dets["persons"],
            "best_dets":  best_dets,
            "best_frame": best_frame,
            "n_persons_total": len(all_persons),
            "n_faces_total":   len(all_faces),
        }

    # ── ffmpeg Helpers ─────────────────────────────────────────────────────────

    def _stop_ffmpeg(self, proc, label="ffmpeg"):
        if proc.poll() is not None:
            return
        try:
            proc.stdin.write(b"q\n")
            proc.stdin.flush()
            proc.wait(timeout=self._cfg.FFMPEG_TIMEOUT)
            log.debug("%s sauber gestoppt", label)
        except subprocess.TimeoutExpired:
            log.warning("%s reagiert nicht – SIGTERM", label)
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                log.warning("%s – SIGKILL", label)
                proc.kill()
                proc.wait()
        except Exception as e:
            log.error("%s Stop-Fehler: %s", label, e)


# ══════════════════════════════════════════════════════════════════════════════════
# StreamMonitor
# ══════════════════════════════════════════════════════════════════════════════════

class StreamMonitor:
    """
    Liest kontinuierlich den Sub-Stream (RTSP), führt YOLO durch
    und triggert ClipRecorder bei Personen-Erkennung.
    """

    def __init__(self, cfg: Config, models: ModelManager, recorder: ClipRecorder):
        self._cfg      = cfg
        self._models   = models
        self._recorder = recorder
        self._running  = False
        self._frame_idx = 0

    def run(self):
        self._running = True
        log.info("Sub : %s", self._cfg.SUB_RTSP)
        log.info("Main: %s", self._cfg.MAIN_RTSP)
        log.info("Out : %s  clip=%ds  cooldown=%ds",
                 self._cfg.OUTPUT_BASE,
                 self._cfg.CLIP_DURATION,
                 self._cfg.CLIP_COOLDOWN)

        while self._running:
            cap = cv2.VideoCapture(self._cfg.SUB_RTSP, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,  5_000)

            if not cap.isOpened():
                log.warning("Sub-Stream nicht verfügbar – Retry in %ds",
                            self._cfg.RECONNECT_DELAY)
                time.sleep(self._cfg.RECONNECT_DELAY)
                continue

            log.info("Sub-Stream offen")

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame-Lesefehler – reconnecting")
                    break

                self._frame_idx += 1
                if self._frame_idx % self._cfg.FRAME_SKIP != 0:
                    continue

                persons = self._models.detect_persons(frame)
                if not persons:
                    continue

                # YOLO-Person = schneller Watchdog → sofort Aufnahme starten.
                # (Schneller als YOLO+InsightFace; der Annähernde ist beim Main-Start
                #  noch im Bild.) Gesichts-Erkennung + Embedding macht Phase 2 am Clip.
                max_conf = max(p["conf"] for p in persons)
                log.info("Person(en): %d  max_conf=%.2f – Trigger", len(persons), max_conf)
                self._recorder.trigger()

            cap.release()
            if self._running:
                log.info("Reconnecting in %ds…", self._cfg.RECONNECT_DELAY)
                time.sleep(self._cfg.RECONNECT_DELAY)

    def stop(self):
        self._running = False


# ══════════════════════════════════════════════════════════════════════════════════
# Cam2App  –  Einstiegspunkt, verdrahtet alle Komponenten
# ══════════════════════════════════════════════════════════════════════════════════

class Cam2App:
    """Haupt-Applikation: Initialisierung, Signal-Handling, Startup/Shutdown."""

    def __init__(self):
        self._cfg      = Config()
        self._models   = ModelManager(self._cfg)
        self._db       = Database(self._cfg.DB)
        self._recorder = ClipRecorder(self._cfg, self._models, self._db)
        self._monitor  = StreamMonitor(self._cfg, self._models, self._recorder)

    def run(self):
        log.info("=== cam2_stream start ===")

        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT,  self._handle_signal)

        self._models.load()
        self._cleanup_old_raws()

        try:
            self._monitor.run()
        finally:
            self._shutdown()

    def _handle_signal(self, sig, frame):
        log.info("Signal %s – stoppe", sig)
        self._monitor.stop()
        self._recorder.stop_all()

    def _shutdown(self):
        self._recorder.stop_all()
        log.info("=== cam2_stream stop ===")

    def _cleanup_old_raws(self):
        """Löscht *_raw.mp4 und *.debug.txt Dateien älter als RAW_KEEP_DAYS Tage."""
        cutoff = time.time() - self._cfg.RAW_KEEP_DAYS * 86_400
        count  = 0
        patterns = ["*_raw.mp4", "*.debug.txt"]
        for pattern in patterns:
            for f in self._cfg.OUTPUT_BASE.rglob(pattern):
                try:
                    if f.stat().st_mtime < cutoff:
                        f.unlink()
                        log.info("Cleanup: gelöscht %s", f)
                        count += 1
                except Exception as e:
                    log.warning("Cleanup-Fehler für %s: %s", f, e)
        if count:
            log.info("Cleanup: %d Datei(en) gelöscht (älter als %d Tage)",
                     count, self._cfg.RAW_KEEP_DAYS)


# ══════════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    Cam2App().run()
