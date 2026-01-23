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
echo "Ziel: ${SOURCE_BASE}/*${TARGET_SUFFIX}/"
echo "Profile: H.264 $PROFILE, Level $LEVEL"
echo "================================================"

# Berechtigungs-Check
if [ ! -w "$SOURCE_BASE" ]; then
    echo ""
    echo "⚠ WARNUNG: Keine Schreibrechte für $SOURCE_BASE"
    echo ""
    if [ "$EUID" -ne 0 ]; then
        echo "Optionen:"
        echo "  1. Mit sudo ausführen: sudo $0"
        echo "  2. Berechtigungen fixen: sudo chown -R $USER:www-data $SOURCE_BASE"
        echo ""
        read -p "Mit sudo-Rechten fortfahren? (j/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Jj]$ ]]; then
            echo "Abgebrochen."
            exit 1
        fi
    fi
fi

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

    # 3. Zielordner erstellen (mit sudo falls nötig)
    if [ ! -d "$TARGET_DIR" ]; then
        echo "  Erstelle: $TARGET_DIR"

        # Versuche normal zu erstellen
        mkdir -p "$TARGET_DIR" 2>/dev/null

        # Falls fehlgeschlagen, versuche mit sudo
        if [ $? -ne 0 ] && [ "$EUID" -ne 0 ]; then
            echo "  → Keine Berechtigung, verwende sudo..."
            sudo mkdir -p "$TARGET_DIR"
            sudo chown $USER:www-data "$TARGET_DIR"
            sudo chmod 775 "$TARGET_DIR"
        fi

        # Erneut prüfen
        if [ ! -d "$TARGET_DIR" ]; then
            echo "✗ FEHLER: Ordner konnte nicht erstellt werden"
            ERRORS=$((ERRORS + 1))
            continue
        fi
    fi

    # 3b. Prüfe Schreibrechte im Zielordner
    if [ ! -w "$TARGET_DIR" ]; then
        echo "  → Keine Schreibrechte, fixe Berechtigungen..."
        if [ "$EUID" -ne 0 ]; then
            sudo chown $USER:www-data "$TARGET_DIR"
            sudo chmod 775 "$TARGET_DIR"
        else
            chown web1:www-data "$TARGET_DIR"
            chmod 775 "$TARGET_DIR"
        fi
    fi

    # 4. FFmpeg-Konvertierung mit Firefox-optimierten Einstellungen
    # CPU-Dekodierung + GPU-Enkodierung (robuster für HEVC-Inputs)
    ffmpeg -i "$input_file" \
      -vf "scale=1920:-2" \
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
        # Rechte setzen (mit aktuellem User oder web1)
        if [ "$EUID" -eq 0 ]; then
            chown web1:www-data "$TARGET_FILE" 2>/dev/null
        else
            chown $USER:www-data "$TARGET_FILE" 2>/dev/null || sudo chown $USER:www-data "$TARGET_FILE" 2>/dev/null
        fi
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
