#!/usr/bin/python3
"""
ebox1arg.py - Enhanced Version
Kommuniziert mit ebox über Serial und speichert Daten automatisch in DB
Ruft pv_ebox2.py mit den empfangenen Daten auf

Usage: ./ebox1arg_enhanced.py <command>
Example: ./ebox1arg_enhanced.py "bat 1"
"""

import serial
import time
import sys
import subprocess
import re

# Konfiguration
SERIAL_PORT = '/dev/ttyUSB23'
BAUD_RATE = 115200
TIMEOUT = 4
PV_EBOX2_SCRIPT = '/home/pi/python/pv_ebox2.py'

def parse_data_line(line):
    """
    Parst eine Datenzeile und extrahiert die 13 Felder (Batterienummer + 12 Datenfelder)

    Erwartet Format (Beispiel):
    b'1     52923  -3342  19000  15000  16000  3305   3311   Dischg   Normal   Normal   Normal   84%\r\n'

    Die Batterienummer wird als "Power" in der DB gespeichert (historische Kompatibilität).
    Die ebox-Hardware gibt kein separates "Power"-Feld aus.

    Returns:
        Tuple (battery_num, data_fields) oder (None, None) wenn ungültig
    """
    try:
        # Bytes zu String
        line_str = line.decode('utf-8', errors='ignore').strip()

        # Überspringe leere Zeilen, Marker und Header
        if not line_str or line_str.startswith('$') or line_str.startswith('#') or 'Power' in line_str:
            return None, None

        # Split nach Whitespace
        parts = line_str.split()

        # Prüfe ob genug Felder vorhanden (mindestens 13: Batterienummer + 12 Datenfelder)
        if len(parts) < 13:
            return None, None

        # Batterienummer und 12 Datenfelder extrahieren (Batterienummer = "Power")
        battery_num = int(parts[0])
        data = parts[0:13]  # Inkl. Batterienummer als erstes Feld

        # Überspringe "Absent" Batterien
        if parts[8] == 'Absent':  # BaseSt ist Feld 8 (absoluter Index)
            return None, None

        # Validierung: Felder sollten numerisch sein
        try:
            int(data[0])    # Batterienummer
            float(data[1])  # Volt
            float(data[2])  # Curr
            return battery_num, data
        except ValueError:
            return None, None

    except Exception as e:
        print(f"Fehler beim Parsen: {e}")
        return None, None

def send_command_and_collect(cmd):
    """
    Sendet Kommando über Serial und sammelt Daten

    Returns:
        Liste mit Datenzeilen oder None bei Fehler
    """
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

        data_lines = []
        cnt = 0
        cmd_bytes = (cmd + "\n").encode()

        print(f"Sende Kommando: {cmd}")

        while True:
            ser.write(cmd_bytes)
            cnt += 1
            data = ser.readline()

            if data:
                print(f"[{cnt}] {data}")
                data_lines.append(data)

                # Ende-Marker erkannt
                if data == b'\r$$\r\n':
                    break

            # Timeout nach 10 Versuchen
            if cnt > 10:
                print("⚠ Timeout: Keine Antwort vom Gerät")
                break

        ser.close()
        return data_lines

    except serial.SerialException as e:
        print(f"✗ Serial-Fehler: {e}")
        return None
    except Exception as e:
        print(f"✗ Fehler: {e}")
        return None

def save_to_database(data_fields):
    """
    Ruft pv_ebox2.py mit den Datenfeldern auf

    Args:
        data_fields: Liste mit 13 Feldern (Batterienummer + 12 Datenfelder)
    """
    try:
        # Baue Kommando
        cmd = [PV_EBOX2_SCRIPT] + data_fields

        print(f"  Volt={data_fields[1]}mV, Curr={data_fields[2]}mA, SOC={data_fields[12]}")

        # Führe Script aus
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("✓ Daten erfolgreich in DB gespeichert")
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")
        else:
            print(f"✗ DB-Insert fehlgeschlagen (Exit Code: {result.returncode})")
            if result.stderr:
                print(f"  Error: {result.stderr.strip()}")

    except subprocess.TimeoutExpired:
        print("✗ Timeout beim DB-Insert")
    except FileNotFoundError:
        print(f"✗ Script nicht gefunden: {PV_EBOX2_SCRIPT}")
    except Exception as e:
        print(f"✗ Fehler beim DB-Insert: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: ./ebox1arg_enhanced.py <command>")
        print("Example: ./ebox1arg_enhanced.py 'bat 1'")
        sys.exit(1)

    cmd = sys.argv[1]

    # 1. Daten vom Gerät holen
    data_lines = send_command_and_collect(cmd)

    if not data_lines:
        print("✗ Keine Daten empfangen")
        sys.exit(1)

    # 2. ALLE Batteriezeilen finden und parsen
    batteries_found = []
    for line in data_lines:
        battery_num, data_fields = parse_data_line(line)
        if battery_num is not None and data_fields is not None:
            batteries_found.append((battery_num, data_fields))

    if not batteries_found:
        print("⚠ Keine gültigen Daten gefunden")
        print("Empfangene Zeilen:")
        for line in data_lines:
            print(f"  {line}")
        sys.exit(1)

    # 3. Alle Batterien in Datenbank speichern
    print(f"\n✓ {len(batteries_found)} Batterie(n) gefunden")
    for battery_num, data_fields in batteries_found:
        print(f"\n→ Verarbeite Batterie {battery_num}")
        save_to_database(data_fields)

    print("\n✓ Fertig!")

if __name__ == "__main__":
    main()
