#!/usr/bin/env python3
"""kfz_districts2db.py - German vehicle registration district codes into wagodb.dim_kfz_district.

Source: de.wikipedia "Liste der Kfz-Kennzeichen in Deutschland" (action=raw wikitext).
Table rows: code | area (Kreis/Stadt) | origin (Herleitung) | state (Bundesland), with rowspans:
one code can cover several areas (A = Stadt Augsburg + Landkreis Augsburg) -> one row per area.

    python3 kfz_districts2db.py --file raw.txt --dry-run   # parse + show
    python3 kfz_districts2db.py                            # fetch, parse, replace table contents
"""
import argparse
import html
import os
import re
import sys
import urllib.request

import pymysql

URL = "https://de.wikipedia.org/w/index.php?title=Liste_der_Kfz-Kennzeichen_in_Deutschland&action=raw"
DDL = """CREATE TABLE IF NOT EXISTS dim_kfz_district (
  id INT AUTO_INCREMENT PRIMARY KEY,
  code VARCHAR(3) NOT NULL COMMENT 'district code as on the plate, with umlauts (TÖL)',
  code_ascii VARCHAR(3) NOT NULL COMMENT 'code as the OCR reads it (umlauts folded: TOL)',
  area VARCHAR(160) NOT NULL COMMENT 'Kreis / kreisfreie Stadt / authority',
  origin VARCHAR(255) NULL COMMENT 'derivation of the code (Herleitung)',
  state VARCHAR(64) NULL COMMENT 'Bundesland',
  source VARCHAR(255) NOT NULL,
  loaded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY ix_code (code),
  KEY ix_code_ascii (code_ascii)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin
  COMMENT='German plate district codes (de.wikipedia list), used by plate_vote.py / rep_kfz.py'"""


def clean(cell):
    c = re.sub(r"<ref[^>]*/>", "", cell)
    c = re.sub(r"<ref[^>]*>.*?</ref>", "", c)
    c = re.sub(r"\{\{[^}]*\}\}", "", c)
    c = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", c)
    c = re.sub(r"\[\[([^\]]*)\]\]", r"\1", c)
    c = re.sub(r"'''|''", "", c)
    c = re.sub(r"<br\s*/?>", " / ", c)
    c = re.sub(r"<[^>]+>", "", c)
    return re.sub(r"\s+", " ", html.unescape(c).replace("\xa0", " ")).strip()


def split_cell(raw):
    """'rowspan="2" | text' -> (2, 'text'); plain -> (1, text)."""
    m = re.match(r'\s*((?:[a-z]+="[^"]*"\s*)+)\|(.*)$', raw)
    if m and "|" not in m.group(1):
        span = re.search(r'rowspan="(\d+)"', m.group(1))
        return (int(span.group(1)) if span else 1), m.group(2)
    return 1, raw


def parse(text):
    rows, cur = [], None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("{|"):
            pending = {}          # column -> (remaining rows, value)
            cur = None
            continue
        if s.startswith("|}"):
            if cur is not None:
                rows.append(cur)
            cur = None
            continue
        if s == "|-":
            if cur is not None:
                rows.append(cur)
            cur = []
            continue
        if cur is None or not s.startswith("|") or s.startswith("|+"):
            continue
        for part in s[1:].split("||"):
            cur.append(part)
    # second pass with rowspans: rebuild per table is approximated by tracking spans globally
    out, spans = [], {}
    for cells in rows:
        full, ci, it = [], 0, iter(cells)
        while len(full) < 4:
            if ci in spans and spans[ci][0] > 0:
                full.append(spans[ci][1]); spans[ci] = (spans[ci][0] - 1, spans[ci][1]); ci += 1
                continue
            try:
                raw = next(it)
            except StopIteration:
                break
            n, val = split_cell(raw)
            val = clean(val)
            if n > 1:
                spans[ci] = (n - 1, val)
            full.append(val); ci += 1
        if len(full) == 4 and re.fullmatch(r"[A-ZÄÖÜ]{1,3}", full[0]):
            out.append(tuple(full))
    return out


def connect():
    return pymysql.connect(read_default_file=os.path.expanduser("~/.my.cnf"),
                           read_default_group="wagodb", connect_timeout=8, charset="utf8mb4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="local wikitext instead of fetching")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.file:
        text = open(a.file, encoding="utf-8").read()
    else:
        req = urllib.request.Request(URL, headers={"User-Agent": "kfz-district-loader/1.0"})
        text = urllib.request.urlopen(req, timeout=40).read().decode("utf-8")
    rows = parse(text)
    codes = {r[0] for r in rows}
    print(f"{len(rows)} rows, {len(codes)} codes")
    if a.dry_run:
        for r in rows[:6] + [r for r in rows if r[0] in ("TÖL", "BN", "M", "RO")]:
            print("  " + " | ".join(r))
        return 0
    fold = str.maketrans("ÄÖÜ", "AOU")
    con = connect()
    with con, con.cursor() as cur:
        cur.execute(DDL)
        cur.execute("DELETE FROM dim_kfz_district")
        cur.executemany(
            "INSERT INTO dim_kfz_district (code, code_ascii, area, origin, state, source) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            [(c, c.translate(fold), area[:160], (orig or None) and orig[:255], st or None,
              "de.wikipedia Liste der Kfz-Kennzeichen in Deutschland") for c, area, orig, st in rows])
        con.commit()
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT code) FROM dim_kfz_district")
        print("in DB: rows=%s codes=%s" % cur.fetchone())
    return 0


if __name__ == "__main__":
    sys.exit(main())
