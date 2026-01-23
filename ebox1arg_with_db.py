#!/usr/bin/python3
"""
ebox1arg_with_db.py - Automatische DB-Speicherung
Basierend auf ebox1arg.py, ruft automatisch pv_ebox2.py auf
"""

import serial
import sys
import subprocess
import os

# Pfad zum DB-Script
PV_EBOX2_SCRIPT = os.path.join(os.path.dirname(__file__), 'pv_ebox2.py')

def main():
    if len(sys.argv) < 2:
        print("Usage: ./ebox1arg_with_db.py <command>")
        sys.exit(1)

    ser = serial.Serial('/dev/ttyUSB23', 115200, timeout=4)
    cnt = 0
    data = ''
    cmd = sys.argv[1] + "\n"

    print(f"Kommando: {cmd.strip()}")

    data_lines = []

    while data != b'\r$$\r\n':
        ser.write(cmd.encode())
        cnt = cnt + 1
        data = ser.readline()
        if data:
            print(data)
            data_lines.append(data)

    ser.close()

    # Suche nach einer Datenzeile mit 13 Feldern
    for line in data_lines:
        try:
            # Decode und split
            line_str = line.decode('utf-8', errors='ignore').strip()

            # Überspringe Marker und leere Zeilen
            if not line_str or line_str.startswith('$') or line_str.startswith('#'):
                continue

            # Split nach Whitespace
            parts = line_str.split()

            # Prüfe ob genug Felder (mindestens 13)
            if len(parts) >= 13:
                # Versuche erste Felder als Zahlen zu parsen (Validierung)
                try:
                    float(parts[0])  # Power
                    float(parts[1])  # Volt

                    # Daten gefunden! Rufe pv_ebox2.py auf
                    data_fields = parts[:13]
                    print(f"\n→ Speichere in DB: {' '.join(data_fields)}")

                    result = subprocess.run(
                        [PV_EBOX2_SCRIPT] + data_fields,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    if result.returncode == 0:
                        print("✓ Daten in DB gespeichert")
                    else:
                        print(f"✗ DB-Fehler: {result.stderr}")

                    break  # Nur erste gültige Zeile verarbeiten

                except ValueError:
                    continue  # Keine gültigen numerischen Daten

        except Exception as e:
            continue  # Nächste Zeile versuchen

    print("✓ Fertig")

if __name__ == "__main__":
    main()
