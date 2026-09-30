#!/usr/bin/env python3
"""
cams_ai.py - cam1 + cam2 in EINEM Prozess (ein CUDA-Kontext auf der Tesla P4).

Ersetzt cam1-ai.service (cam1_daemon.py) + cam2-ai.service (cam2_streamOO.py).
Keine eigene AI-/DB-Logik: haengt nur an person.py, cam1_daemon.py, cam2_streamOO.py.

Geteilt:
  - CUDA-Kontext (Torch + onnxruntime)
  - InsightFace buffalo_l (beide prepare(det_size=640, det_thresh=0.3) -> identisch)
  - EIN Lock um jeden GPU-Aufruf (cam2 Live-Stream wartet max. eine Inferenz)
NICHT geteilt:
  - YOLO: ultralytics 8.0.235 merkt sich Predictor-Args ueber Aufrufe hinweg
    (get_cfg(self.predictor.args, args)) -> cam2 classes=[0] wuerde cam1 die
    Fahrzeuge nehmen, cam1 half=True wuerde cam2 auf FP16 stellen.
    Zwei YOLO-Instanzen kosten nur ~100 MB.
  - Known-Faces + Match-Schwellen: jede Kamera behaelt ihre eigene Logik
    (cam2 inkl. .npy-Banken, FACE_THRESHOLD 0.45).

Faellt der cam1-Thread aus, beendet sich der Prozess -> systemd startet neu.
"""
import logging
import logging.handlers
import os
import sys
import threading

import cam2_streamOO as cam2   # konfiguriert Root-Logging beim Import (wird unten ersetzt)
import person                  # dito (basicConfig dann No-op)
import cam1_daemon

log = logging.getLogger("cams_ai")


# ── Logging: cam1 und cam2 wieder in ihre bisherigen Dateien trennen ─────────────

def _setup_logging():
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    root.addHandler(stdout)  # Rest (ultralytics, insightface, cams_ai) -> Journal

    def attach(names, fmt, handlers):
        f = logging.Formatter(fmt)
        for h in handlers:
            h.setFormatter(f)
        for n in names:
            lg = logging.getLogger(n)
            lg.propagate = True           # zusaetzlich ins Journal (root/stdout)
            for h in handlers:
                lg.addHandler(h)

    attach(["person", "cam1_daemon"],
           "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
           [logging.FileHandler("/home/gh/python/logs/reolink_processor.log"),
            logging.handlers.RotatingFileHandler("/tmp/cam1-ai.log",
                                                 maxBytes=10 * 1024 * 1024, backupCount=1)])
    attach([cam2.__name__],
           "%(asctime)s [cam2] %(levelname)s %(message)s",
           [logging.FileHandler(cam2.Config.LOG_FILE),
            logging.handlers.RotatingFileHandler("/tmp/cam2-ai.log",
                                                 maxBytes=10 * 1024 * 1024, backupCount=1)])


# ── GPU-Lock-Proxys fuer cam1 (person.AIAnalyzer ruft die Modelle direkt auf) ─────

class _LockedYolo:
    """yolo_model(...) unter Lock; .names usw. durchgereicht."""
    def __init__(self, model, lock):
        self._m, self._lock = model, lock

    def __call__(self, *a, **kw):
        with self._lock:
            return self._m(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._m, name)


class _LockedFaceApp:
    """face_app.get(...) unter Lock; Rest durchgereicht."""
    def __init__(self, app, lock):
        self._a, self._lock = app, lock

    def get(self, *a, **kw):
        with self._lock:
            return self._a.get(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._a, name)


class _LockedAlpr:
    def __init__(self, alpr, lock):
        self._a, self._lock = alpr, lock

    def predict(self, *a, **kw):
        with self._lock:
            return self._a.predict(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._a, name)


# ── cam2 mit geteiltem InsightFace + Lock ────────────────────────────────────────

class SharedModelManager(cam2.ModelManager):
    def __init__(self, cfg, face_app, lock):
        super().__init__(cfg)
        self._shared_face_app = face_app
        self._lock = lock  # ersetzt den eigenen Lock -> gemeinsam mit cam1

    def _load_insightface(self):
        self._face_app = self._shared_face_app
        log.info("cam2: InsightFace geteilt mit cam1 (%s)",
                 "aktiv" if self._face_app is not None else "FEHLT")


class ComboCam2App(cam2.Cam2App):
    def __init__(self, models, on_loaded):
        self._cfg      = models._cfg
        self._models   = models
        self._db       = cam2.Database(self._cfg.DB)
        self._recorder = cam2.ClipRecorder(self._cfg, self._models, self._db)
        self._monitor  = cam2.StreamMonitor(self._cfg, self._models, self._recorder)
        # cam1 erst starten, wenn cam2 fertig geladen hat (Known-Faces ohne Lock)
        orig_load = self._models.load

        def load_then_start():
            orig_load()
            on_loaded()
        self._models.load = load_then_start


def _cam1_thread(ai):
    try:
        cam1_daemon.run_cam1(ai_analyzer=ai)
        log.error("cam1-Thread beendet")
    except BaseException:
        log.exception("cam1-Thread abgestuerzt")
    finally:
        logging.shutdown()
        os._exit(1)  # ganzer Dienst neu -> systemd Restart=always


def main():
    _setup_logging()
    log.info("=== cams_ai start (cam1 + cam2, ein CUDA-Kontext) ===")
    lock = threading.RLock()

    ai = cam1_daemon.build_analyzer()  # YOLO(cam1) + InsightFace + ALPR + Known-Faces
    raw_face_app = ai.face_app
    ai.yolo_model = _LockedYolo(ai.yolo_model, lock)
    if ai.face_app is not None:
        ai.face_app = _LockedFaceApp(ai.face_app, lock)
    if getattr(ai, "alpr", None) is not None:
        ai.alpr = _LockedAlpr(ai.alpr, lock)

    cfg = cam2.Config()
    models = SharedModelManager(cfg, raw_face_app, lock)

    def start_cam1():
        threading.Thread(target=_cam1_thread, args=(ai,), name="cam1", daemon=True).start()
        log.info("cam1-Thread gestartet")

    ComboCam2App(models, start_cam1).run()
    log.info("=== cams_ai stop ===")


if __name__ == "__main__":
    main()
