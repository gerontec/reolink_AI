#!/usr/bin/env python3
"""vehicle_track.py - follow every vehicle through a cam1 clip and say what it did.

Independent of the licence plate: a car that only drives along the road is a pass even when
the OCR reads a few characters, a car that turns off the road onto the property is an entry.

Zones come from cam1_zones.json next to this file (full-resolution pixel coordinates):
    {"image_size": [4512, 2512], "road": [[x, y], ...]}
Everything that is not road counts as property. A vehicle's position is the bottom centre of
its box (where the wheels touch the ground).

    track(dets, fps)  dets: [{frame, bbox (x1,y1,x2,y2), cls, conf}] for one clip
                      -> list of tracks: {track_no, vehicle_class, motion, direction,
                         start_zone, end_zone, first_s, last_s, n_points, max_box_w, frames}
    motion: pass (stays on the road and moves) / entry (road -> property) /
            exit (property -> road) / yard_move (moves on the property) / parked /
            road_stop (on the road, not moving)

Self test: python3 vehicle_track.py
"""
import json
import os

ZONES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cam1_zones.json")
MAX_GAP_S = 2.0        # a track may miss detections for this long
MIN_POINTS = 2


def load_zones(path=ZONES_FILE):
    with open(path, encoding="utf-8") as f:
        z = json.load(f)
    return [tuple(p) for p in z["road"]]


def inside(pt, poly):
    """Ray casting point-in-polygon."""
    x, y = pt
    n, hit = len(poly), False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xc:
                hit = not hit
    return hit


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _foot(b):
    return ((b[0] + b[2]) / 2, b[3])


def associate(dets, fps):
    """Greedy frame-by-frame association (IoU, else centre distance < 0.6 box width)."""
    max_gap = max(1, int(MAX_GAP_S * (fps or 20)))
    tracks = []
    by_frame = {}
    for d in dets:
        by_frame.setdefault(d["frame"], []).append(d)
    for fr in sorted(by_frame):
        live = [t for t in tracks if fr - t[-1]["frame"] <= max_gap]
        used = set()
        for d in sorted(by_frame[fr], key=lambda d: -d.get("conf", 0)):
            best, score = None, 0.0
            for ti, t in enumerate(live):
                if ti in used:
                    continue
                last = t[-1]["bbox"]
                s = _iou(last, d["bbox"])
                if s < 0.2:
                    (ax, ay), (bx, by) = _centre(last), _centre(d["bbox"])
                    w = max(last[2] - last[0], 1)
                    dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                    s = 0.1 * (1 - dist / (0.6 * w)) if dist < 0.6 * w else 0.0
                if s > score:
                    best, score = ti, s
            if best is None:
                tracks.append([d])
                live.append(tracks[-1])
                used.add(len(live) - 1)
            else:
                live[best].append(d)
                used.add(best)
    return tracks


def classify(t, road, fps):
    feet = [_foot(d["bbox"]) for d in t]
    on_road = [inside(p, road) for p in feet]
    widths = sorted(d["bbox"][2] - d["bbox"][0] for d in t)
    w_med = widths[len(widths) // 2]
    k = max(1, len(t) // 5)                     # first / last 20 % decide start and end zone
    start_road = sum(on_road[:k]) * 2 >= k
    end_road = sum(on_road[-k:]) * 2 >= k
    dx = feet[-1][0] - feet[0][0]
    dy = feet[-1][1] - feet[0][1]
    moved = max(abs(dx), abs(dy)) > 0.5 * w_med or \
        max(p[0] for p in feet) - min(p[0] for p in feet) > 0.5 * w_med
    road_share = sum(on_road) / len(on_road)
    if start_road and end_road:
        motion = "pass" if moved and road_share >= 0.6 else ("road_stop" if not moved else "pass")
    elif start_road and not end_road:
        motion = "entry"
    elif not start_road and end_road:
        motion = "exit"
    else:
        motion = "yard_move" if moved else "parked"
    classes = {}
    for d in t:
        classes[d.get("cls", "vehicle")] = classes.get(d.get("cls", "vehicle"), 0) + 1
    fs = fps or 20
    return {"vehicle_class": max(classes, key=classes.get), "motion": motion,
            "direction": ("ltr" if dx > 0 else "rtl") if moved else None,
            "start_zone": "road" if start_road else "property",
            "end_zone": "road" if end_road else "property",
            "first_s": round(t[0]["frame"] / fs, 1), "last_s": round(t[-1]["frame"] / fs, 1),
            "n_points": len(t), "max_box_w": int(widths[-1]),
            "frames": (t[0]["frame"], t[-1]["frame"])}


def track(dets, fps, road=None):
    road = road or load_zones()
    out = []
    for t in associate(dets, fps):
        if len(t) < MIN_POINTS:
            continue
        c = classify(t, road, fps)
        c["bboxes"] = [(d["frame"], d["bbox"]) for d in t]
        out.append(c)
    out.sort(key=lambda c: c["frames"][0])
    for i, c in enumerate(out, 1):
        c["track_no"] = i
    return out


def plate_for_track(tr, reads):
    """Readings whose car box belongs to this track (same frame, IoU >= 0.5)."""
    boxes = dict(tr["bboxes"])
    return [r for r in reads if r.get("frame") in boxes and _iou(boxes[r["frame"]], r["bbox"]) >= 0.5]


if __name__ == "__main__":
    road = load_zones()
    # synthetic: a car along the road (left -> right), one turning onto the yard, one parked
    along = [{"frame": f, "bbox": (1200 + 60 * f, 700 - 25 * f - 150, 1500 + 60 * f, 700 - 25 * f), "cls": "car"}
             for f in range(0, 20)]
    turn = [{"frame": 100 + f, "bbox": (2300 - 10 * f, 300 + 40 * f - 200, 2600 - 10 * f, 300 + 40 * f), "cls": "car"}
            for f in range(0, 20)]
    park = [{"frame": f, "bbox": (3630, 140, 4140, 530), "cls": "car"} for f in range(0, 200, 5)]
    for tr in track(along + turn + park, 20, road):
        print(tr["track_no"], tr["motion"], tr["direction"], tr["start_zone"], "->", tr["end_zone"],
              f"{tr['first_s']}-{tr['last_s']}s", tr["n_points"])
