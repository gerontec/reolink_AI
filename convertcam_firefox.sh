#!/bin/bash

# --- KONFIGURATION ---
SOURCE_BASE="/var/www/web1"
TARGET_SUFFIX="mime"

# Firefox-kompatible Einstellungen
BITRATE="2500k"
MAXRATE="3000k"
BUFSIZE="5000k"
PROFILE="main"          # main oder baseline für maximale Kompatibilität
LEVEL="4.1"             # Level für Full HD
PRESET="p4"             # NVENC Preset (p1-p7, p4 = balanced)

# --- LOGIK ---

echo "================================================"
echo "Video-Konverter für Firefox-Kompatibilität"
echo "Quelle: $SOURCE_BASE"
echo "Ziel: ${SOURCE_BASE}/${YEAR}${TARGET_SUFFIX}/"
echo "Profile: H.264 $PROFILE, Level $LEVEL"
echo "================================================"
echo ""

# Zähler
TOTAL=0
CONVERTED=0
SKIPPED=0
ERRORS=0

# Suche alle MP4 Dateien
find "$SOURCE_BASE" -regextype posix-extended -regex ".*/[0-9]{4}/[0-9]{2}/.*\.mp4" -type f | while read -r input_file; do

    TOTAL=$((TOTAL + 1))

    # Pfad-Teile extrahieren
    YEAR=$(echo "$input_file" | cut -d'/' -f5)
    MONTH=$(echo "$input_file" | cut -d'/' -f6)
    FILENAME=$(basename "$input_file")

    # Ziel-Struktur
    TARGET_DIR="/var/www/web1/${YEAR}${TARGET_SUFFIX}/${MONTH}"
    TARGET_FILE="${TARGET_DIR}/${FILENAME}"

    # 1. Überspringen falls existiert
    if [ -f "$TARGET_FILE" ]; then
        echo "⊘ Existiert bereits: $FILENAME"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # 2. Quelldatei prüfen
    if [ ! -f "$input_file" ]; then
        echo "✗ Quelldatei nicht gefunden: $input_file"
        ERRORS=$((ERRORS + 1))
        continue
    fi

    echo ""
    echo "→ Verarbeite: $FILENAME"
    echo "  Quelle: $input_file"
    echo "  Ziel: $TARGET_FILE"

    # 3. Zielordner erstellen
    if [ ! -d "$TARGET_DIR" ]; then
        echo "  Erstelle: $TARGET_DIR"
        mkdir -p "$TARGET_DIR" 2>/dev/null

        if [ $? -ne 0 ]; then
            echo "✗ FEHLER: Ordner konnte nicht erstellt werden (Rechte?)"
            ERRORS=$((ERRORS + 1))
            continue
        fi
    fi

    # 4. FFmpeg-Konvertierung mit Firefox-optimierten Einstellungen
    ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i "$input_file" \
      -vf "scale_cuda=1920:-2" \
      -c:v h264_nvenc \
      -preset "$PRESET" \
      -profile:v "$PROFILE" \
      -level "$LEVEL" \
      -b:v "$BITRATE" \
      -maxrate "$MAXRATE" \
      -bufsize "$BUFSIZE" \
      -pix_fmt yuv420p \
      -c:a aac \
      -b:a 128k \
      -ar 48000 \
      -ac 2 \
      -movflags +faststart \
      -f mp4 \
      -y "$TARGET_FILE" \
      2>&1 | grep -v "frame=" | grep -v "time=" | tail -5

    # 5. Prüfen und Rechte setzen
    if [ -f "$TARGET_FILE" ] && [ -s "$TARGET_FILE" ]; then
        # Rechte setzen
        chown web1:www-data "$TARGET_FILE" 2>/dev/null
        chmod 644 "$TARGET_FILE" 2>/dev/null

        # Dateigröße prüfen
        ORIG_SIZE=$(stat -f%z "$input_file" 2>/dev/null || stat -c%s "$input_file")
        NEW_SIZE=$(stat -f%z "$TARGET_FILE" 2>/dev/null || stat -c%s "$TARGET_FILE")
        ORIG_MB=$((ORIG_SIZE / 1024 / 1024))
        NEW_MB=$((NEW_SIZE / 1024 / 1024))

        echo "✓ Erfolgreich: $TARGET_FILE"
        echo "  Original: ${ORIG_MB}MB → Neu: ${NEW_MB}MB"
        CONVERTED=$((CONVERTED + 1))
    else
        echo "✗ FEHLER: FFmpeg konnte Datei nicht schreiben oder Datei ist leer"
        # Lösche fehlerhafte Datei
        rm -f "$TARGET_FILE" 2>/dev/null
        ERRORS=$((ERRORS + 1))
    fi

done

echo ""
echo "================================================"
echo "Konvertierung abgeschlossen"
echo "================================================"
echo "Gesamt:       $TOTAL Videos gefunden"
echo "Konvertiert:  $CONVERTED Videos"
echo "Übersprungen: $SKIPPED Videos (bereits vorhanden)"
echo "Fehler:       $ERRORS Videos"
echo "================================================"
