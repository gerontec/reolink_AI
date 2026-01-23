#!/usr/bin/env python3
"""
pv_ebox2.py - KORRIGIERTE VERSION
Speichert ebox Daten in MariaDB Tabelle pv_ebox2

Aufruf: ./pv_ebox2.py Power Volt Curr Tempr Tlow Thigh Vlow Vhigh BaseSt VoltSt CurrSt TempSt Coulomb
Beispiel: ./pv_ebox2.py 1 52923 -3342 19000 15000 16000 3305 3311 Dischg Normal Normal Normal 84%
"""

import sys
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

# Konfiguration
DB_URL = "mysql+pymysql://gh:a12345@192.168.178.218/wagodb"

def process_and_save(raw_args):
    """
    Verarbeitet Kommandozeilenargumente und speichert in DB

    Args:
        raw_args: Liste mit 13 Datenfeldern (ohne Batterie-Nummer)
    """
    # Check auf Mindestlänge (13 Datenfelder)
    if not raw_args or len(raw_args) < 13 or raw_args[0] == '-':
        print(f"⚠ Ungültige Daten (benötigt 13 Felder, erhalten: {len(raw_args)})")
        return

    # Spaltennamen exakt wie in deiner SQL-Tabelle (ohne id)
    cols = [
        "Power", "Volt", "Curr", "Tempr", "Tlow", "Thigh",
        "Vlow", "Vhigh", "BaseSt", "VoltSt", "CurrSt", "TempSt", "Coulomb"
    ]

    # Daten extrahieren und % bei Coulomb (SOC) entfernen
    data = list(raw_args[:13])
    data[12] = data[12].replace('%', '')

    # DataFrame erstellen
    df = pd.DataFrame([data], columns=cols)

    # Numerische Typen wandeln
    num_cols = ["Power", "Volt", "Curr", "Tempr", "Tlow", "Thigh", "Vlow", "Vhigh", "Coulomb"]

    try:
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print(f"⚠ Warnung beim Konvertieren numerischer Felder: {e}")

    # Zeitstempel als echtes Datetime-Objekt
    df['ts'] = datetime.now()

    # In MariaDB schreiben (KORREKTUR: verwende engine.connect())
    try:
        engine = create_engine(DB_URL)

        # WICHTIG: Verwende with-Statement für Connection
        with engine.connect() as connection:
            df.to_sql('pv_ebox2', con=connection, if_exists='append', index=False)

        print(f"✓ Daten in DB gespeichert: Power={data[0]}W, Volt={data[1]}mV, SOC={data[12]}")

    except Exception as e:
        print(f"✗ Fehler beim DB-Insert: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./pv_ebox2.py Power Volt Curr Tempr Tlow Thigh Vlow Vhigh BaseSt VoltSt CurrSt TempSt Coulomb")
        print("Example: ./pv_ebox2.py 1 52923 -3342 19000 15000 16000 3305 3311 Dischg Normal Normal Normal 84%")
        sys.exit(1)

    process_and_save(sys.argv[1:])
