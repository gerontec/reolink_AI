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

# NEU: Verarbeite ALLE Batteriezeilen (1, 2, 3)
for line in all_lines:
    try:
        line_str = line.decode('utf-8', errors='ignore').strip()

        # Überspringe leere Zeilen, Marker und Header
        if not line_str or '$' in line_str or '#' in line_str or 'Power' in line_str:
            continue

        parts = line_str.split()

        # Prüfe ob mindestens 14 Felder vorhanden (Batterienummer + 13 Daten)
        if len(parts) >= 14:
            # Validiere dass Felder numerisch sind
            try:
                bat_num = int(parts[0])  # Batterienummer
                float(parts[1])  # Power
                float(parts[2])  # Volt

                # Überspringe "Absent" Batterien
                if parts[8] == 'Absent':
                    continue

                # Daten gefunden! Verwende Felder 1-13 (ohne Batterienummer)
                data_fields = parts[1:14]
                print(f"\n→ Batterie {bat_num}: {' '.join(data_fields)}")
                subprocess.run(['/home/pi/python/pv_ebox2.py'] + data_fields, timeout=5)

                # KEIN break hier - verarbeite alle Batterien!

            except ValueError:
                continue
    except:
        continue
