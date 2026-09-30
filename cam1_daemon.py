#!/usr/bin/env python3
"""
cam1_daemon.py - Persistenter cam1-Verarbeitungs-Daemon.

Ersetzt den stuendlichen Cron (run_chain.sh -> person.py): laedt YOLO+InsightFace
EINMAL (person.AIAnalyzer) und verarbeitet neue FTP-Uploads laufend per
mtime-Wasserzeichen -> KEINE Modell-Cold-Starts mehr und kein DB-SELECT-Sturm
(nur wirklich neue Dateien werden geprueft).

Wiederverwendung statt Duplikat: haengt NUR an person.py (AIAnalyzer,
FileProcessor.process_file, Konstanten). Keine eigene AI-/DB-Logik.

(b)-freundlich (spaeterer Single-Service cam1+cam2):
  - run_cam1(ai_analyzer=...) nimmt den schweren Analyzer per Dependency Injection.
    Ein vereinter Service erzeugt EINEN AIAnalyzer und startet run_cam1(shared) in
    einem Thread, waehrend cam2 denselben Analyzer nutzt -> ein Modellsatz, ein
    CUDA-Kontext. Bei geteiltem Analyzer die GPU-Aufrufe serialisieren (Lock).
"""
import time
import logging
from datetime import datetime
from pathlib import Path

import person  # AIAnalyzer, FileProcessor, Konstanten, DB_CONFIG

POLL_INTERVAL   = 20      # s zwischen Scans
SETTLE_SECONDS  = 5       # Datei muss so alt sein, bevor verarbeitet (FTP fertig)
STARTUP_LOOKBACK = 2 * 3600  # beim Start Backlog der letzten 2h aufholen (Cron-Luecke)
BATCH_LIMIT     = 500     # max. Dateien pro Durchlauf

log = logging.getLogger("cam1_daemon")


def build_analyzer():
    """Erzeugt den (teuren) AIAnalyzer - einmalig."""
    return person.AIAnalyzer(
        person.YOLO_MODEL_PATH,
        person.KNOWN_FACES_DIR,
        force_gpu=True,
    )


def run_cam1(ai_analyzer=None, poll_interval: int = POLL_INTERVAL):
    """cam1-Verarbeitungsschleife.

    ai_analyzer: vorgeladener person.AIAnalyzer (DI fuer spaeteren Single-Service);
                 None -> selbst laden (Standalone-Daemon).
    """
    if ai_analyzer is None:
        log.info("Lade AIAnalyzer (YOLO+InsightFace) einmalig ...")
        ai_analyzer = build_analyzer()
    log.info("cam1-Daemon bereit (Device=%s, %d Known-Faces). Poll=%ss Settle=%ss",
             ai_analyzer.device, len(ai_analyzer.known_face_names),
             poll_interval, SETTLE_SECONDS)

    # Wasserzeichen: nur Dateien neuer als dieses mtime werden betrachtet.
    # Start 2h in der Vergangenheit -> Backlog seit letztem Cron-Lauf aufholen
    # (bereits analysierte werden von process_file via DB ohnehin geskippt).
    watermark = time.time() - STARTUP_LOOKBACK

    while True:
        try:
            month_dir = Path(person.MEDIA_BASE_PATH) / datetime.now().strftime("%Y/%m")
            now = time.time()
            new = []
            if month_dir.exists():
                for ext in ("*.mp4", "*.jpg"):
                    for p in month_dir.rglob(ext):
                        try:
                            m = p.stat().st_mtime
                        except OSError:
                            continue
                        if watermark < m <= now - SETTLE_SECONDS:
                            new.append((m, p))
            new.sort()  # aelteste zuerst

            if new:
                processor = person.FileProcessor(
                    str(month_dir), person.DB_CONFIG, ai_analyzer,
                    person.ANNOTATED_OUTPUT_PATH, force=False,
                )
                if not processor.connect_db():
                    log.error("Keine DB-Verbindung - Durchlauf uebersprungen")
                    time.sleep(poll_interval)
                    continue
                try:
                    handled = 0
                    for m, p in new:
                        try:
                            processor.process_file(p, analyze=True)  # dedupt intern via DB
                        except Exception:
                            log.exception("Datei-Fehler: %s", p.name)
                        watermark = max(watermark, m)
                        handled += 1
                        if handled >= BATCH_LIMIT:
                            break
                    log.info("cam1: %d Kandidat(en) -> verarbeitet=%d analysiert=%d "
                             "annotiert=%d skip=%d fehler=%d",
                             handled, processor.processed_count, processor.analyzed_count,
                             processor.annotated_count, processor.skipped_count,
                             processor.error_count)
                finally:
                    processor.disconnect_db()
        except Exception:
            log.exception("cam1-Durchlauf fehlgeschlagen")
        time.sleep(poll_interval)


if __name__ == "__main__":
    run_cam1()
