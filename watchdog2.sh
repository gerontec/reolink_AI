# Mit Face-Cropping (nur bekannte Gesichter)
./watchdog2.py --save-face-crops --save-annotated --limit 10

# Produktiv ohne Limit
./watchdog2.py --save-face-crops --save-annotated

# Nur Analyse, keine Speicherung
./watchdog2.py

# CPU-Modus für Tests
./watchdog2.py --cpu-only
