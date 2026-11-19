#!/usr/bin/env python3
"""plate_vote.py - German plate format check + multi-frame voting for the cam1 ALPR.

The OCR models of fast-plate-ocr know only 0-9 A-Z (no umlauts), and a clip gives one reading
per frame. Instead of keeping only the single most confident reading, every reading of a clip
is parsed against the German plate format and the readings are voted:

  parse_de(raw)  -> best interpretation of one OCR text as a German plate, or None
                    district (1-3 letters, from plate_districts_de.txt, umlauts restored)
                    + 1-2 letters + 1-4 digits (no leading 0) + optional E/H suffix.
                    OCR confusions are repaired per segment (digit<->letter, O/G/D/U/Q),
                    each repair costs 1; at most MAX_COST repairs per reading.
  vote(reads)    -> winner over all readings of one clip, weighted conf * 0.5**cost.

Plate text is stored without separators, as before (TÖLMF582); `display` gives TÖL-MF 582.
District list: de.wikipedia "Liste der Kfz-Kennzeichen in Deutschland" (732 codes, 19.11.2026).

Self test:  python3 plate_vote.py TOLMF582 TGLMF582 10LWF582 BNR2456E GB6473 OLFF22
"""
import os
import sys
from collections import defaultdict

DISTRICT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plate_districts_de.txt")
MAX_COST = 2

FOLD = str.maketrans("ÄÖÜ", "AOU")
# OCR reading -> letter it may stand for (cost 1)
AS_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "7": "T", "8": "B"}
# OCR reading -> digit it may stand for (cost 1)
AS_DIGIT = {"O": "0", "D": "0", "Q": "0", "U": "0", "G": "6", "I": "1", "L": "1", "T": "1",
            "J": "1", "Z": "2", "S": "5", "B": "8", "A": "4"}
# letters the OCR swaps among each other (district matching only, cost 1)
LETTER_SWAP = [set("OGDUQ0"), set("TI1"), set("L1I"), set("BE8"), set("MW"), set("CG")]
# what the OCR (alphabet without umlauts) reads for an umlaut in the district, cost 0.5;
# the dots make Ö look like 8/B/6 (seen 19.11.: T8L, TBL, T6L for TÖL)
UMLAUT_READS = {"Ö": set("O0QDUG86B"), "Ü": set("U0OVW"), "Ä": set("A4")}


def _load_districts():
    with open(DISTRICT_FILE, encoding="utf-8") as f:
        codes = [c.strip() for c in f if c.strip()]
    by_fold = defaultdict(list)            # "TOL" -> ["TÖL"]; "GO" etc. may hold 2 codes
    for c in codes:
        by_fold[c.translate(FOLD)].append(c)
    return by_fold


DISTRICTS = _load_districts()
DISTRICTS_BY_LEN = defaultdict(list)
for _f in DISTRICTS:
    DISTRICTS_BY_LEN[len(_f)].append(_f)


def _char_cost(read, want):
    """Cost of OCR char `read` standing for district char `want` (may be an umlaut), None if impossible."""
    if want in UMLAUT_READS:
        # an umlaut can never be read exactly, so half a repair only
        return 0.5 if read in UMLAUT_READS[want] else None
    if read == want:
        return 0
    if AS_LETTER.get(read) == want:
        return 1
    if any(read in s and want in s for s in LETTER_SWAP):
        return 1
    return None


def _district(seg):
    """Cheapest district code for an OCR segment -> (cost, code) or None.

    An umlaut can never be read exactly (not in the OCR alphabet), so it costs 0.5;
    on equal cost a plain-ASCII code wins over an umlaut code.
    """
    best = None
    for fold in DISTRICTS_BY_LEN[len(seg)]:
        for code in DISTRICTS[fold]:
            cost = 0
            for r, w in zip(seg, code):
                c = _char_cost(r, w)
                if c is None:
                    break
                cost += c
            else:
                key = (cost, code != fold)
                if best is None or key < best[0]:
                    best = (key, code)
    return (best[0][0], best[1]) if best else None


def _letters(seg):
    out, cost = [], 0
    for ch in seg:
        if ch.isalpha():
            out.append(ch)
        elif ch in AS_LETTER:
            out.append(AS_LETTER[ch]); cost += 1
        else:
            return None
    return cost, "".join(out)


def _digits(seg):
    out, cost = [], 0
    for ch in seg:
        if ch.isdigit():
            out.append(ch)
        elif ch in AS_DIGIT:
            out.append(AS_DIGIT[ch]); cost += 1
        else:
            return None
    if out[0] == "0":
        return None
    return cost, "".join(out)


def parse_de(raw):
    """-> dict(text, display, district, cost) for the cheapest German reading of `raw`, or None."""
    raw = "".join(ch for ch in (raw or "").upper() if ch.isalnum())
    n = len(raw)
    best = None
    for suffix in ("", "E", "H"):
        if suffix and not raw.endswith(suffix):
            continue
        body = raw[:len(raw) - len(suffix)] if suffix else raw
        m = len(body)
        for d in (1, 2, 3):
            for l in (1, 2):
                k = m - d - l
                if not 1 <= k <= 4 or d + l + k > 8:
                    continue
                dist = _district(body[:d])
                if dist is None:
                    continue
                let = _letters(body[d:d + l])
                if let is None:
                    continue
                dig = _digits(body[d + l:])
                if dig is None:
                    continue
                cost = dist[0] + let[0] + dig[0]
                if cost > MAX_COST:
                    continue
                cand = {"text": dist[1] + let[1] + dig[1] + suffix,
                        "display": f"{dist[1]}-{let[1]} {dig[1]}{suffix}",
                        "district": dist[1], "cost": cost}
                # cheapest wins; on a tie the longer district (more specific) wins
                if best is None or (cost, -d) < (best["cost"], -len(best["district"])):
                    best = cand
    if n == 0:
        return None
    return best


def vote(reads):
    """reads: iterable of dicts with at least 'text' and 'conf' (one per OCR reading of a clip).

    -> dict(text, display, district, conf, n_reads, n_valid, vote_share, raw_text, format_ok)
       or None if `reads` is empty. Without any valid German reading the most confident raw
       reading is returned with format_ok=False (old behaviour).
    """
    reads = list(reads)
    if not reads:
        return None
    best_raw = max(reads, key=lambda r: r["conf"])
    score = defaultdict(float)
    info, best_conf, best_raw_of = {}, defaultdict(float), {}
    n_valid = 0
    for r in reads:
        p = parse_de(r["text"])
        if p is None:
            continue
        n_valid += 1
        w = r["conf"] * (0.5 ** p["cost"])
        score[p["text"]] += w
        info[p["text"]] = p
        if r["conf"] > best_conf[p["text"]]:
            best_conf[p["text"]] = r["conf"]
            best_raw_of[p["text"]] = r
    if not score:
        return {"text": best_raw["text"], "display": best_raw["text"], "district": None,
                "conf": best_raw["conf"], "n_reads": len(reads), "n_valid": 0,
                "vote_share": 0.0, "raw_text": best_raw["text"], "format_ok": False,
                "read": best_raw}
    win = max(score, key=lambda t: (score[t], best_conf[t]))
    return {"text": win, "display": info[win]["display"], "district": info[win]["district"],
            "conf": best_conf[win], "n_reads": len(reads), "n_valid": n_valid,
            "vote_share": round(score[win] / sum(score.values()), 3),
            "raw_text": best_raw_of[win]["text"], "format_ok": True,
            "read": best_raw_of[win]}


def _quality_pool(reads, width_frac=0.8, min_aspect=2.0):
    """Readings taken while the plate was large and upright in the image.

    Parked/rotated plates give a near-square box (84x64) and garbage; the approach phase gives
    wide flat boxes (97x31). Falls back to the width filter alone, then to all readings.
    """
    sized = [r for r in reads if r.get("plate_w") and r.get("plate_h")]
    if not sized:
        return list(reads)
    wmax = max(r["plate_w"] for r in sized)
    wide = [r for r in sized if r["plate_w"] >= width_frac * wmax]
    flat = [r for r in wide if r["plate_w"] / r["plate_h"] >= min_aspect]
    return flat or wide


def fuse(reads, k=3):
    """Character-wise fusion of the k best readings of one clip.

    1. pool = readings with a large, upright plate (_quality_pool)
    2. top k of the pool by OCR confidence
    3. most common text length among them; per position the char with the highest summed
       per-character probability wins (falls back to the reading confidence per char)
    4. the fused text is checked against the German format; if it is not valid, the best
       valid single reading of the top k is taken; without any, the most confident raw text.
    -> same dict as vote() plus 'top' (the k readings used)
    """
    reads = list(reads)
    if not reads:
        return None
    top = sorted(_quality_pool(reads), key=lambda r: r["conf"], reverse=True)[:k]
    lengths = defaultdict(float)
    for r in top:
        lengths[len(r["text"])] += r["conf"]
    n = max(lengths, key=lengths.get)
    same = [r for r in top if len(r["text"]) == n]
    fused = []
    for i in range(n):
        score = defaultdict(float)
        for r in same:
            cc = r.get("char_conf")
            score[r["text"][i]] += cc[i] if cc and i < len(cc) else r["conf"]
        fused.append(max(score, key=score.get))
    fused = "".join(fused)
    conf = sum(r["conf"] for r in same) / len(same)
    base = {"n_reads": len(reads), "n_valid": len(top), "vote_share": round(len(same) / len(top), 3),
            "fused_raw": fused, "top": top}
    p = parse_de(fused)
    if p is None:
        cands = [(parse_de(r["text"]), r) for r in top]
        cands = [(q, r) for q, r in cands if q]
        if cands:
            p, r = min(cands, key=lambda qr: (qr[0]["cost"], -qr[1]["conf"]))
            conf = r["conf"]
    if p is None:
        best = top[0]
        return dict(base, text=best["text"], display=best["text"], district=None, conf=best["conf"],
                    raw_text=best["text"], format_ok=False, read=best)
    return dict(base, text=p["text"], display=p["display"], district=p["district"], conf=conf,
                raw_text=fused, format_ok=True, read=same[0])


def motion(reads, fps):
    """What the vehicle with the plate did during the clip, from its readings over time.

    Uses the car box centre per reading (frame coordinates) and the frame number:
      * the longest run of readings whose centre stays within `tol` of the run start is the
        STOP (tol = a quarter of the median car box width, so it scales with distance)
      * movement before the stop -> the car drove in   (arrival)
      * movement after the stop  -> the car drove off  (departure)
      * both -> arrival_departure (short stop inside one clip), neither -> parked,
        no stop at all -> pass (drove through)
    -> dict(motion, first_seen_s, last_seen_s, event_s) or None without frame numbers.
       event_s = arrival: start of the stop; departure: end of the stop; else first sighting.
    """
    rs = sorted((r for r in reads if r.get("frame") is not None and r.get("bbox")), key=lambda r: r["frame"])
    if not rs or not fps:
        return None
    cx = [(r["bbox"][0] + r["bbox"][2]) / 2 for r in rs]
    cy = [(r["bbox"][1] + r["bbox"][3]) / 2 for r in rs]
    widths = sorted(r["bbox"][2] - r["bbox"][0] for r in rs)
    tol = 0.25 * widths[len(widths) // 2]
    best = (0, 0, -1)                       # (length, start, end) of the longest stop run
    i = 0
    while i < len(rs):
        j = i
        while j + 1 < len(rs) and abs(cx[j + 1] - cx[i]) <= tol and abs(cy[j + 1] - cy[i]) <= tol:
            j += 1
        if j - i + 1 > best[0]:
            best = (j - i + 1, i, j)
        i = j + 1
    n, s, e = best
    sec = lambda k: round(rs[k]["frame"] / fps, 1)
    first, last = sec(0), sec(len(rs) - 1)
    span = rs[-1]["frame"] - rs[0]["frame"]
    stop_frames = rs[e]["frame"] - rs[s]["frame"]
    if n < 3 or (span and stop_frames < 0.3 * span):
        return {"motion": "pass", "first_seen_s": first, "last_seen_s": last, "event_s": first}

    def moved(a, b):
        return any(abs(cx[k] - cx[b]) > tol or abs(cy[k] - cy[b]) > tol for k in a)
    before = moved(range(0, s), s) if s > 0 else False
    after = moved(range(e + 1, len(rs)), e) if e < len(rs) - 1 else False
    if before and after:
        m, ev = "arrival_departure", sec(s)
    elif before:
        m, ev = "arrival", sec(s)
    elif after:
        m, ev = "departure", sec(e)
    else:
        m, ev = "parked", first
    return {"motion": m, "first_seen_s": first, "last_seen_s": last, "event_s": ev}


if __name__ == "__main__":
    for t in sys.argv[1:] or ["TOLMF582", "TGLMF582", "T0LMF582", "T8LMF582", "TBLMF582", "T01MF582",
                              "10LWF582", "10LMF582", "BNR2456E", "GB6473", "OLFF22", "AA2307JA"]:
        p = parse_de(t)
        print(f"{t:<10} -> " + (f"{p['display']}  (cost {p['cost']})" if p else "no German plate"))
