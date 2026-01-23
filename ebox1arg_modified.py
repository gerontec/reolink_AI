#!/usr/bin/python3
# ebox.py bat 1
# ebox.py pwr 1
# MODIFIZIERT: Ruft automatisch pv_ebox2.py auf um Daten in DB zu speichern

import serial
import time
import sys
import subprocess

ser = serial.Serial('/dev/ttyUSB23', 115200, timeout=4)
cnt = 0
data = ''
cmd = sys.argv[1]+"\n"
print(cmd)

# Liste um Datenzeilen zu sammeln
all_lines = []

while data != b'\r$$\r\n':
    ser.write(cmd.encode())
    cnt = cnt+1
    data = ser.readline()
    if data:
        print(data)
        all_lines.append(data)  # Sammle alle Zeilen

ser.close()

# NEU: Versuche Datenzeile zu finden und in DB zu speichern
for line in all_lines:
    try:
        line_str = line.decode('utf-8', errors='ignore').strip()

        # Überspringe leere Zeilen und Marker
        if not line_str or '$' in line_str or '#' in line_str:
            continue

        parts = line_str.split()

        # Prüfe ob mindestens 13 Felder vorhanden
        if len(parts) >= 13:
            # Validiere dass erste Felder numerisch sind
            try:
                float(parts[0])
                float(parts[1])

                # Daten gefunden! Rufe pv_ebox2.py auf
                print(f"\n→ Rufe pv_ebox2.py auf mit: {' '.join(parts[:13])}")
                subprocess.run(['/home/pi/python/pv_ebox2.py'] + parts[:13], timeout=5)
                break

            except ValueError:
                continue
    except:
        continue
