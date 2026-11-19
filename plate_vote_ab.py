#!/usr/bin/env python3
"""plate_vote_ab.py - A/B test: old "most confident reading" vs. plate_vote on the SAME readings.

For each cam1 clip that already has a kfz_kennzeichen row and is still on disk, every frame is
searched for plates exactly like person.AIAnalyzer does it in production (YOLO vehicle boxes,
500 px tiles, OCR >= 0.60, det >= 0.25, len >= 5). From that one set of readings both methods
pick their result, so the only difference is the selection (one change per test).
Also records the plate box size (pixels) for the later camera comparison.
No DB writes. Output: TSV to stdout-file + summary.

    run_kfz.sh-style env needed (LD_LIBRARY_PATH for CUDA), see run_plate_ab.sh
    python3 plate_vote_ab.py [--n 40] [--step 2] [--out /mnt/archive/plate_ab.tsv]
"""
import argparse
import glob
import os
import re
import statistics
import sys
import time
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort
import pymysql
from fast_alpr import ALPR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plate_vote  # noqa: E402
import vehicle_track  # noqa: E402

P = ["CUDAExecutionProvider", "CPUExecutionProvider"]
VEH_COLS = [2, 3, 5, 7]          # COCO car, motorcycle, bus, truck (as kfz_alpr_v2.py)
YOLO_CONF = 0.40


def connect():
    return pymysql.connect(read_default_file=os.path.expanduser("~/.my.cnf"),
                           read_default_group="wagodb", connect_timeout=8, charset="utf8mb4")


def letterbox(img, s=640):
    h, w = img.shape[:2]; r = min(s / w, s / h); nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((s, s, 3), 114, np.uint8); px, py = (s - nw) // 2, (s - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return canvas, r, px, py


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]); ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(boxes, th=0.5):
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True); keep = []
    while boxes:
        b = boxes.pop(0); keep.append(b); boxes = [o for o in boxes if iou(b, o) < th]
    return keep


class Reader:
    def __init__(self):
        self.yolo = ort.InferenceSession("/opt/models/yolov8m.onnx", providers=P)
        self.yin = self.yolo.get_inputs()[0].name
        self.alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end", detector_providers=P,
                         ocr_model="european-plates-mobile-vit-v2-model", ocr_providers=P)

    def cars(self, frame):
        lb, r, px, py = letterbox(frame)
        blob = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        out = self.yolo.run(None, {self.yin: blob})[0][0].T
        vc = out[:, 4:][:, VEH_COLS].max(axis=1); keep = vc >= YOLO_CONF
        boxes = []
        for p, c in zip(out[keep], vc[keep]):
            cx, cy, bw, bh = p[:4]
            boxes.append([(cx - bw / 2 - px) / r, (cy - bh / 2 - py) / r,
                          (cx + bw / 2 - px) / r, (cy + bh / 2 - py) / r, float(c)])
        return nms(boxes)

    def plates(self, frame, cars):
        """All readings in this frame, same tiling/thresholds as person._detect_plate."""
        def cv(c):
            if c is None:
                return 0.0
            if isinstance(c, (list, tuple)):
                v = [float(x) for x in c if x is not None]; return sum(v) / len(v) if v else 0.0
            return float(c)
        H, W = frame.shape[:2]; reads = []
        for b in cars:
            x1, y1 = int(max(0, b[0])), int(max(0, b[1])); x2, y2 = int(min(W, b[2])), int(min(H, b[3]))
            car = frame[y1:y2, x1:x2]
            if car.size == 0:
                continue
            ch, cw = car.shape[:2]; cols = max(1, round(cw / 500)); rows = max(1, round(ch / 500)); ov = 0.25
            tw, th = cw // cols, ch // rows
            for ty in range(rows):
                for tx in range(cols):
                    a0, b0 = max(0, int(tx * tw - ov * tw)), max(0, int(ty * th - ov * th))
                    a1, b1 = min(cw, int((tx + 1) * tw + ov * tw)), min(ch, int((ty + 1) * th + ov * th))
                    try:
                        preds = self.alpr.predict(car[b0:b1, a0:a1])
                    except Exception:
                        continue
                    for r in preds:
                        o = getattr(r, "ocr", None); d = getattr(r, "detection", None)
                        if not o or not getattr(o, "text", ""):
                            continue
                        oc = cv(getattr(o, "confidence", 0)); dc = cv(getattr(d, "confidence", 0))
                        txt = re.sub(r"[^A-Z0-9]", "", (o.text or "").upper())
                        if oc < 0.60 or dc < 0.25 or len(txt) < 5:
                            continue
                        bb = getattr(d, "bounding_box", None)
                        pw = (bb.x2 - bb.x1) if bb else None; ph = (bb.y2 - bb.y1) if bb else None
                        raw_cc = getattr(o, "confidence", None)
                        cc = [float(x) for x in raw_cc] if isinstance(raw_cc, (list, tuple)) else None
                        reads.append({"text": txt, "conf": oc, "plate_w": pw, "plate_h": ph, "char_conf": cc,
                                      "bbox": (x1, y1, x2, y2)})
        return reads


def clips(n):
    con = connect()
    with con, con.cursor() as cur:
        cur.execute("""SELECT k.recording_id, k.plate_text, r.file_path
                       FROM kfz_kennzeichen k JOIN cam2_recordings r ON r.id = k.recording_id
                       ORDER BY k.recording_id DESC""")
        rows = cur.fetchall()
    out = []
    for rid, txt, fp in rows:
        p = os.path.join("/var/www/web1", fp)
        if not os.path.isfile(p):
            m = glob.glob(f"/var/www/web1/20*/*/{os.path.basename(fp)}")
            p = m[0] if m else None
        if p:
            out.append((rid, txt, p))
        if len(out) >= n:
            break
    return out


def clips_by_rid(rids):
    """Any cam1 recording by id (also without a kfz row)."""
    con = connect()
    with con, con.cursor() as cur:
        cur.execute("SELECT r.id, (SELECT k.plate_text FROM kfz_kennzeichen k WHERE k.recording_id=r.id LIMIT 1),"
                    " r.file_path FROM cam2_recordings r WHERE r.id IN (" + ",".join(["%s"] * len(rids)) + ")",
                    list(rids))
        rows = cur.fetchall()
    out = []
    for rid, txt, fp in rows:
        p = os.path.join("/var/www/web1", fp)
        if not os.path.isfile(p):
            m = glob.glob(f"/var/www/web1/20*/*/{os.path.basename(fp)}")
            p = m[0] if m else None
        if p:
            out.append((rid, txt or "", p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--step", type=int, default=2, help="every n-th frame (production: 1 once a car is seen)")
    ap.add_argument("--out", default="/mnt/archive/plate_ab.tsv")
    ap.add_argument("--rid", type=int, action="append", help="only these recording ids")
    ap.add_argument("--show-reads", action="store_true", help="print every reading with frame number")
    ap.add_argument("--update-motion", action="store_true",
                    help="write motion/first_seen_s/last_seen_s/event_s (+ n_reads, plate size) into the "
                         "existing kfz_kennzeichen rows; plate_text stays untouched")
    a = ap.parse_args()
    rd = Reader()
    if a.rid:
        todo = clips_by_rid(a.rid)
    else:
        todo = clips(a.n)
    print(f"{len(todo)} clips on disk", flush=True)
    res = []
    with open(a.out, "w") as f:
        f.write("recording_id\tdb_text\told_text\told_conf\tnew_text\tnew_display\tformat_ok\t"
                "vote_share\tn_reads\tn_valid\tplate_w_med\tplate_h_med\tfuse_text\tfuse_display\tfuse_raw\tfuse_plate_w\n")
        for i, (rid, dbtxt, path) in enumerate(todo, 1):
            t0 = time.time(); cap = cv2.VideoCapture(path); fn = 0; reads = []; dets = []
            fps = cap.get(cv2.CAP_PROP_FPS) or None
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                fn += 1
                if fn % a.step:
                    continue
                c = rd.cars(fr)
                dets += [{"frame": fn, "bbox": tuple(int(v) for v in b[:4]), "cls": "vehicle", "conf": b[4]} for b in c]
                if c:
                    for r in rd.plates(fr, c):
                        r["frame"] = fn
                        reads.append(r)
            cap.release()
            old = max(reads, key=lambda r: r["conf"]) if reads else None
            new = plate_vote.vote(reads)
            fu = plate_vote.fuse(reads)
            mo = plate_vote.motion(reads, fps) or {}
            tracks = vehicle_track.track(dets, fps)
            if a.update_motion and reads:
                w = [r["plate_w"] for r in fu["top"] if r.get("plate_w")] if fu else []
                h = [r["plate_h"] for r in fu["top"] if r.get("plate_h")] if fu else []
                con = connect()
                with con, con.cursor() as cur:
                    cur.execute("UPDATE kfz_kennzeichen SET motion=%s, first_seen_s=%s, last_seen_s=%s, event_s=%s,"
                                " n_reads=COALESCE(n_reads,%s), plate_w=COALESCE(plate_w,%s), plate_h=COALESCE(plate_h,%s)"
                                " WHERE recording_id=%s",
                                (mo.get("motion"), mo.get("first_seen_s"), mo.get("last_seen_s"), mo.get("event_s"),
                                 len(reads), int(statistics.median(w)) if w else None,
                                 int(statistics.median(h)) if h else None, rid))
                    con.commit()
            ws = [r["plate_w"] for r in reads if r["plate_w"]]; hs = [r["plate_h"] for r in reads if r["plate_h"]]
            row = [rid, dbtxt, old["text"] if old else "", f"{old['conf']:.2f}" if old else "",
                   new["text"] if new else "", new["display"] if new else "",
                   int(new["format_ok"]) if new else "", new["vote_share"] if new else "",
                   len(reads), new["n_valid"] if new else 0,
                   int(statistics.median(ws)) if ws else "", int(statistics.median(hs)) if hs else "",
                   fu["text"] if fu else "", fu["display"] if fu else "", fu["fused_raw"] if fu else "",
                   int(statistics.median([r["plate_w"] for r in fu["top"] if r["plate_w"]] or [0])) if fu else ""]
            f.write("\t".join(map(str, row)) + "\n"); f.flush()
            for tr in tracks:
                pr = vehicle_track.plate_for_track(tr, reads)
                print(f"    track {tr['track_no']}: {tr['motion']:<9} {tr['direction'] or '-':<3} "
                      f"{tr['start_zone']}->{tr['end_zone']} {tr['first_s']}-{tr['last_s']}s "
                      f"pts={tr['n_points']} box<={tr['max_box_w']}px plate_reads={len(pr)}"
                      + (f" ({max(pr, key=lambda r: r['conf'])['text']})" if pr else ""), flush=True)
            if a.show_reads:
                for r in reads:
                    p = plate_vote.parse_de(r["text"])
                    print(f"    frame {r['frame']:4d}  {r['text']:<10} conf {r['conf']:.3f}  "
                          f"plate {r['plate_w']}x{r['plate_h']}px  -> {p['display'] if p else '-'}")
            res.append(row)
            print(f"[{i}/{len(todo)}] rec {rid}: db={dbtxt} old={row[2]} vote={row[5]} fuse={row[13]} "
                  f"motion={mo.get('motion')} {mo.get('first_seen_s')}-{mo.get('last_seen_s')}s ev={mo.get('event_s')}s "
                  f"reads={len(reads)} share={row[7]} plate={row[10]}x{row[11]}px ({time.time() - t0:.0f}s)",
                  flush=True)
    old_t = Counter(r[2] for r in res if r[2]); new_t = Counter(r[4] for r in res if r[4])
    fu_t = Counter(r[12] for r in res if r[12])
    print("\nSUMMARY")
    print(f"  clips with readings     {sum(1 for r in res if r[8])}/{len(res)}")
    print(f"  distinct texts old/vote/fuse  {len(old_t)} / {len(new_t)} / {len(fu_t)}")
    print(f"  valid German format new {sum(1 for r in res if r[6] == 1)}")
    print("  top old: " + ", ".join(f"{t} {n}" for t, n in old_t.most_common(6)))
    print("  top vote: " + ", ".join(f"{t} {n}" for t, n in new_t.most_common(6)))
    print("  top fuse: " + ", ".join(f"{t} {n}" for t, n in fu_t.most_common(6)))


if __name__ == "__main__":
    main()
