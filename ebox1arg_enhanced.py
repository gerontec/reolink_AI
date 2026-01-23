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
    Parst eine Datenzeile und extrahiert die 13 Felder

    Erwartet Format (Beispiel):
    b'123.4 48.2 2.5 25.3 20.1 30.5 45.0 50.0 OK OK OK OK 85%\r\n'

    Returns:
        Liste mit 13 Werten oder None wenn ungültig
    """
    try:
        # Bytes zu String
        line_str = line.decode('utf-8', errors='ignore').strip()

        # Überspringe leere Zeilen und Marker
        if not line_str or line_str.startswith('$') or line_str.startswith('#'):
            return None

        # Split nach Whitespace
        parts = line_str.split()

        # Prüfe ob genug Felder vorhanden (mindestens 13)
        if len(parts) < 13:
            return None

        # Erste 13 Felder extrahieren
        data = parts[:13]

        # Validierung: Erste Felder sollten numerisch sein
        try:
            float(data[0])  # Power
            float(data[1])  # Volt
            float(data[2])  # Curr
            return data
        except ValueError:
            return None

    except Exception as e:
        print(f"Fehler beim Parsen: {e}")
        return None

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
        data_fields: Liste mit 13 Datenfeldern
    """
    try:
        # Baue Kommando
        cmd = [PV_EBOX2_SCRIPT] + data_fields

        print(f"\n→ Speichere in DB: {' '.join(data_fields)}")

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

    # 2. Relevante Datenzeile finden und parsen
    valid_data = None
    for line in data_lines:
        parsed = parse_data_line(line)
        if parsed:
            valid_data = parsed
            break

    if not valid_data:
        print("⚠ Keine gültigen Daten gefunden")
        print("Empfangene Zeilen:")
        for line in data_lines:
            print(f"  {line}")
        sys.exit(1)

    # 3. In Datenbank speichern
    save_to_database(valid_data)

    print("\n✓ Fertig!")

if __name__ == "__main__":
    main()
