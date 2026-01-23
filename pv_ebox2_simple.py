#!/usr/bin/env python3
"""
pv_ebox2_simple.py - Einfache Version ohne pandas
Speichert ebox Daten direkt mit pymysql in MariaDB

Aufruf: ./pv_ebox2_simple.py Power Volt Curr Tempr Tlow Thigh Vlow Vhigh BaseSt VoltSt CurrSt TempSt Coulomb
Beispiel: ./pv_ebox2_simple.py 1 52923 -3342 19000 15000 16000 3305 3311 Dischg Normal Normal Normal 84%
"""

import sys
import pymysql
from datetime import datetime

# Konfiguration
DB_CONFIG = {
    'host': '192.168.178.218',
    'database': 'wagodb',
    'user': 'gh',
    'password': 'a12345'
}

def save_to_db(data_fields):
    """
    Speichert Daten direkt in DB (ohne pandas)

    Args:
        data_fields: Liste mit 13 Datenfeldern
    """
    # Validierung
    if not data_fields or len(data_fields) < 13 or data_fields[0] == '-':
        print(f"⚠ Ungültige Daten (benötigt 13 Felder, erhalten: {len(data_fields)})")
        return False

    try:
        # % bei SOC entfernen
        coulomb = data_fields[12].replace('%', '')

        # Numerische Felder konvertieren
        power = float(data_fields[0])
        volt = float(data_fields[1])
        curr = float(data_fields[2])
        tempr = float(data_fields[3])
        tlow = float(data_fields[4])
        thigh = float(data_fields[5])
        vlow = float(data_fields[6])
        vhigh = float(data_fields[7])
        coulomb_num = float(coulomb)

        # String-Felder
        base_st = data_fields[8]
        volt_st = data_fields[9]
        curr_st = data_fields[10]
        temp_st = data_fields[11]

        # Zeitstempel
        ts = datetime.now()

        # DB-Connection
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # INSERT
        query = """
            INSERT INTO pv_ebox2
            (Power, Volt, Curr, Tempr, Tlow, Thigh, Vlow, Vhigh,
             BaseSt, VoltSt, CurrSt, TempSt, Coulomb, ts)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        cursor.execute(query, (
            power, volt, curr, tempr, tlow, thigh, vlow, vhigh,
            base_st, volt_st, curr_st, temp_st, coulomb_num, ts
        ))

        conn.commit()
        cursor.close()
        conn.close()

        print(f"✓ Daten gespeichert: Power={power}W, Volt={volt}mV, SOC={coulomb}%")
        return True

    except ValueError as e:
        print(f"✗ Fehler beim Konvertieren der Daten: {e}")
        return False
    except pymysql.Error as e:
        print(f"✗ DB-Fehler: {e}")
        return False
    except Exception as e:
        print(f"✗ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./pv_ebox2_simple.py Power Volt Curr Tempr Tlow Thigh Vlow Vhigh BaseSt VoltSt CurrSt TempSt Coulomb")
        print("Example: ./pv_ebox2_simple.py 1 52923 -3342 19000 15000 16000 3305 3311 Dischg Normal Normal Normal 84%")
        sys.exit(1)

    success = save_to_db(sys.argv[1:])
    sys.exit(0 if success else 1)
