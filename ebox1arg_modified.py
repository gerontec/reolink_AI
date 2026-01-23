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

        # Prüfe ob mindestens 13 Felder vorhanden (Batterienummer + 12 Datenfelder)
        if len(parts) >= 13:
            # Validiere dass Felder numerisch sind
            try:
                bat_num = int(parts[0])  # Batterienummer
                float(parts[1])  # Volt
                float(parts[2])  # Curr

                # Überspringe "Absent" Batterien
                if parts[8] == 'Absent':
                    continue

                # Daten gefunden! Verwende Felder 0-12 (Batterienummer als "Power" + 12 Datenfelder)
                # Die Batterienummer wird in DB-Spalte "Power" gespeichert (so war es auch vorher)
                data_fields = parts[0:13]
                print(f"\n→ Batterie {bat_num}: Volt={parts[1]}mV, Curr={parts[2]}mA, SOC={parts[12]}")
                subprocess.run(['/home/pi/python/pv_ebox2.py'] + data_fields, timeout=5)

                # KEIN break hier - verarbeite alle Batterien!

            except ValueError:
                continue
    except:
        continue
