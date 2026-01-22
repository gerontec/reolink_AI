# Mit Face-Cropping (nur bekannte Gesichter)
./watchdog_improved.py --save-face-crops --save-annotated --limit 10

# Produktiv ohne Limit
./watchdog_improved.py --save-face-crops --save-annotated

# Nur Analyse, keine Speicherung
./watchdog_improved.py

# CPU-Modus für Tests
./watchdog_improved.py --cpu-only
