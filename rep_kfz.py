#!/usr/bin/env python3
"""rep_kfz.py - report of the licence plates the cam pipeline wrote to MariaDB.

Reads wagodb on the dell (192.168.5.23, credentials from ~/.my.cnf [wagodb]):

  kfz_kennzeichen   one row per plate reading (kfz_alpr_v2.py via cams-ai): plate_text,
                    confidence, bbox, plate_crop_path, detected_at, recording_id
  cam2_recordings   the recording a reading came from (camera_name, recorded_at, file_path)

The OCR reads one physical plate in many spellings (TOLMF582, TGLMF582, T0LMF582, 10LWF582 ...).
Readings are therefore grouped into "vehicles": every text is mapped onto a confusion-tolerant
key (O/Q/D/U/G -> 0, T/I/J -> 1, B -> 8, S -> 5, Z -> 2, W -> M), and keys that differ by at most
one edit (same length class) are merged. The group name is its most frequent spelling.
Known landscape false positives (e.g. GB6473, the plate detector hallucinating in trees) are
listed separately and left out of the vehicle count.
Separate groups at the end: district prefixes (--prefix, default BN = Bonn) and electric vehicles
(E plates: identification number ending in digit + E, any district). Deutsche Post / DHL is assigned
by the BN prefix ONLY - its fleet runs on petrol/diesel too, so an E plate alone says nothing.

Usage:
    python3 rep_kfz.py                 # totals, per day, vehicles
    python3 rep_kfz.py --days 7        # only the last 7 days
    python3 rep_kfz.py --top 50        # longer vehicle list
    python3 rep_kfz.py --raw           # also every distinct text with its count
    python3 rep_kfz.py --min-conf 0.7  # ignore readings below this confidence
    python3 rep_kfz.py --prefix BN --prefix M   # district groups (default BN = Bonn, Post fleet)
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

import pymysql

# texts the plate detector produces on background (trees, houses), see project notes
FALSE_POSITIVES = {"GB6473", "GB6474"}

CONFUSION = str.maketrans({
    "O": "0", "Q": "0", "D": "0", "U": "0", "G": "0",
    "T": "1", "I": "1", "J": "1",
    "B": "8", "S": "5", "Z": "2", "W": "M",
})


def connect():
    return pymysql.connect(read_default_file=os.path.expanduser("~/.my.cnf"),
                           read_default_group="wagodb", connect_timeout=8,
                           charset="utf8mb4", cursorclass=pymysql.cursors.DictCursor)


def fetch(con, days, min_conf):
    where, params = ["1=1"], []
    if days:
        where.append("k.detected_at >= NOW() - INTERVAL %s DAY")
        params.append(days)
    if min_conf:
        where.append("k.confidence >= %s")
        params.append(min_conf)
    sql = f"""
        SELECT k.id, k.plate_text, k.confidence, k.detected_at, k.recording_id,
               r.camera_name, r.recorded_at
        FROM kfz_kennzeichen k
        LEFT JOIN cam2_recordings r ON r.id = k.recording_id
        WHERE {' AND '.join(where)}
        ORDER BY k.detected_at"""
    with con.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def key(text):
    return (text or "").upper().replace(" ", "").replace("-", "").translate(CONFUSION)


def within_one_edit(a, b):
    """True if a and b differ by at most one substitution, insertion or deletion."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    if la > lb:
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def group_texts(counts):
    """Greedy clustering on confusion keys; returns {text: group representative}.

    Keys are visited by frequency; a key joins the first cluster whose CENTRE key (its most
    frequent one) is within one edit, otherwise it opens a new cluster. Comparing against the
    centre only - not against every member - avoids chaining unrelated plates together.
    """
    key_count = Counter()
    for t, n in counts.items():
        key_count[key(t)] += n
    centres = []                      # centre keys, most frequent first
    key_centre = {}
    for k, _ in sorted(key_count.items(), key=lambda kv: (-kv[1], kv[0])):
        # short keys (< 5 chars) are too ambiguous to merge
        hit = next((c for c in centres
                    if min(len(c), len(k)) >= 5 and within_one_edit(c, k)), None)
        if hit is None:
            centres.append(k)
            hit = k
        key_centre[k] = hit

    members = defaultdict(list)
    for t in counts:
        members[key_centre[key(t)]].append(t)
    rep = {}
    for texts in members.values():
        best = max(texts, key=lambda t: (counts[t], t))
        for t in texts:
            rep[t] = best
    return rep


def fmt_ts(t):
    return t.strftime("%d.%m. %H:%M") if t else "-"


def report_prefix(rows, prefix):
    """All readings whose plate starts with the district prefix (e.g. BN = Bonn, Deutsche Post fleet).

    Matched on the raw text and on the confusion key, so OCR slips like 8N... are caught too.
    A trailing E (electric vehicle) is flagged. No merging of spellings here: each text is listed.
    """
    pkey = key(prefix)
    hits = [r for r in rows if (r["plate_text"] or "").upper().startswith(prefix)
            or key(r["plate_text"]).startswith(pkey)]

    def note(t):
        n = []
        if ELECTRIC_RE.search(t.upper()):
            n.append("E = electric")
        if not t.upper().startswith(prefix):
            n.append("OCR variant")
        return ", ".join(n)

    title = f"Prefix {prefix}" + (" = Deutsche Post / DHL" if prefix == "BN" else "")
    print_text_group(title, hits, note)


# German e-plate: the identification number ends in E directly after a digit (e.g. BN R 2456E)
ELECTRIC_RE = re.compile(r"\dE$")


def report_electric(rows):
    """All readings with an E plate (electric vehicle), any district - no Post/DHL assignment here."""
    hits = [r for r in rows if ELECTRIC_RE.search((r["plate_text"] or "").upper())]
    print_text_group("Electric (E plates)", hits,
                     lambda t: f"district {t[:2]}?" if len(t) > 2 else "")


def print_text_group(title, hits, note):
    """One line per distinct text, most readings first, plus readings per day."""
    print(f"\n{title} ({len(hits)} readings, {len({r['plate_text'] for r in hits})} texts)")
    if not hits:
        print("  none")
        return
    by_text = defaultdict(list)
    for r in hits:
        by_text[r["plate_text"]].append(r)
    print(f"  {'plate':<10} {'reads':>5} {'conf':>5} {'days':>4}  {'first':<12} {'last':<12} note")
    for t, rs in sorted(by_text.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        print(f"  {t:<10} {len(rs):5d} {max(r['confidence'] for r in rs):5.2f} "
              f"{len({r['detected_at'].date() for r in rs}):4d}  "
              f"{fmt_ts(rs[0]['detected_at']):<12} {fmt_ts(rs[-1]['detected_at']):<12} {note(t)}")
    days = Counter(r["detected_at"].date() for r in hits)
    print("  per day: " + ", ".join(f"{d:%d.%m.} {n}" for d, n in sorted(days.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, help="only the last N days")
    ap.add_argument("--top", type=int, default=20, help="number of vehicles to list (default 20)")
    ap.add_argument("--min-conf", type=float, default=0.0, help="minimum OCR confidence")
    ap.add_argument("--raw", action="store_true", help="also list every distinct text")
    ap.add_argument("--prefix", action="append", default=None,
                    help="district prefix to group separately, repeatable (default: BN)")
    a = ap.parse_args()
    if a.prefix is None:
        a.prefix = ["BN"]

    try:
        con = connect()
    except pymysql.MySQLError as e:
        print(f"wagodb not reachable: {e}", file=sys.stderr)
        return 1
    with con:
        rows = fetch(con, a.days, a.min_conf)

    scope = f"last {a.days} days" if a.days else "all data"
    if a.min_conf:
        scope += f", confidence >= {a.min_conf}"
    print(f"Licence plates in wagodb.kfz_kennzeichen ({scope})")
    if not rows:
        print("  no readings")
        return 0

    texts = Counter(r["plate_text"] for r in rows)
    fp_rows = [r for r in rows if r["plate_text"] in FALSE_POSITIVES]
    real = [r for r in rows if r["plate_text"] not in FALSE_POSITIVES]
    rep = group_texts(Counter(r["plate_text"] for r in real))

    print(f"  readings        {len(rows)}")
    print(f"  recordings      {len({r['recording_id'] for r in rows})}")
    print(f"  distinct texts  {len(texts)}")
    print(f"  vehicles        {len(set(rep.values()))}   (grouped, without false positives)")
    print(f"  false positives {len(fp_rows)}   ({', '.join(sorted(FALSE_POSITIVES))})")
    print(f"  period          {fmt_ts(rows[0]['detected_at'])} - {fmt_ts(rows[-1]['detected_at'])}")
    cams = Counter(r["camera_name"] or "?" for r in rows)
    print("  per camera      " + ", ".join(f"{c} {n}" for c, n in cams.most_common()))

    print("\nPer day (readings / vehicles)")
    per_day = defaultdict(list)
    for r in real:
        per_day[r["detected_at"].date()].append(rep[r["plate_text"]])
    for d in sorted(per_day):
        v = per_day[d]
        print(f"  {d:%a %d.%m.}  {len(v):4d} / {len(set(v)):3d}  {'#' * min(len(v), 60)}")

    groups = defaultdict(list)
    for r in real:
        groups[rep[r["plate_text"]]].append(r)
    print(f"\nVehicles (top {a.top} by readings)")
    print(f"  {'plate':<10} {'reads':>5} {'conf':>5} {'days':>4}  {'first':<12} {'last':<12} spellings")
    ranked = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    for name, rs in ranked[:a.top]:
        spell = Counter(r["plate_text"] for r in rs)
        others = [t for t, _ in spell.most_common() if t != name]
        more = f" +{len(others) - 4}" if len(others) > 4 else ""
        print(f"  {name:<10} {len(rs):5d} {max(r['confidence'] for r in rs):5.2f} "
              f"{len({r['detected_at'].date() for r in rs}):4d}  "
              f"{fmt_ts(rs[0]['detected_at']):<12} {fmt_ts(rs[-1]['detected_at']):<12} "
              f"{', '.join(others[:4])}{more}")
    singles = sum(1 for rs in groups.values() if len(rs) == 1)
    print(f"  ... {len(groups) - min(a.top, len(groups))} more, of which {singles} seen only once")

    for prefix in a.prefix:
        report_prefix(real, prefix.upper())
    report_electric(real)

    if a.raw:
        print("\nAll texts")
        for t, n in texts.most_common():
            print(f"  {t:<12} {n:4d}  -> {'FALSE POSITIVE' if t in FALSE_POSITIVES else rep[t]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
