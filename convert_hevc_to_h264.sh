#!/bin/bash
#
# HEVC to H.264 Converter für Reolink Videos
# Konvertiert alle HEVC-Videos zu Firefox-kompatiblem H.264
#

set -euo pipefail

# Konfiguration
VIDEO_DIR="/var/www/web1/2026/01"
LOG_FILE="/home/gh/python/logs/video_conversion.log"
TEMP_DIR="/tmp/video_convert"
PARALLEL_JOBS=2  # Anzahl paralleler Konvertierungen

# Erstelle Temp-Verzeichnis
mkdir -p "$TEMP_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Logging-Funktion
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Prüfe ob Video HEVC ist
is_hevc() {
    local video="$1"
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$video" 2>/dev/null | grep -q "hevc"
}

# Konvertiere einzelnes Video
convert_video() {
    local input="$1"
    local basename=$(basename "$input")
    local temp_output="${TEMP_DIR}/${basename%.mp4}_h264.mp4"
    local final_output="${input}.new"

    log "Konvertiere: $basename"

    # Konvertierung mit CPU
    if ffmpeg -y -i "$input" \
        -c:v libx264 -preset medium -crf 23 \
        -c:a copy \
        -movflags +faststart \
        "$temp_output" >> "$LOG_FILE" 2>&1; then

        # Größen vergleichen
        local old_size=$(stat -f%z "$input" 2>/dev/null || stat -c%s "$input")
        local new_size=$(stat -f%z "$temp_output" 2>/dev/null || stat -c%s "$temp_output")
        local old_mb=$((old_size / 1024 / 1024))
        local new_mb=$((new_size / 1024 / 1024))

        # Verschiebe konvertiertes Video
        mv "$temp_output" "$final_output"

        # Ersetze Original (nach erfolgreichem Test)
        if ffprobe -v error "$final_output" >/dev/null 2>&1; then
            rm "$input"
            mv "$final_output" "$input"
            log "✓ Erfolgreich: $basename (${old_mb}MB → ${new_mb}MB)"
        else
            log "✗ Fehler bei Verifikation: $basename"
            rm -f "$final_output"
            return 1
        fi
    else
        log "✗ Konvertierung fehlgeschlagen: $basename"
        rm -f "$temp_output"
        return 1
    fi
}

# Hauptprogramm
main() {
    log "=== Start HEVC zu H.264 Konvertierung ==="
    log "Verzeichnis: $VIDEO_DIR"

    # Finde alle MP4-Videos
    local count=0
    local converted=0
    local skipped=0
    local failed=0

    # Exportiere Funktion für parallel
    export -f convert_video
    export -f log
    export -f is_hevc
    export LOG_FILE
    export TEMP_DIR

    # Sammle HEVC-Videos
    local hevc_videos=()
    while IFS= read -r -d '' video; do
        if is_hevc "$video"; then
            hevc_videos+=("$video")
        else
            ((skipped++))
        fi
    done < <(find "$VIDEO_DIR" -maxdepth 1 -name "*.mp4" -type f -print0 | sort -z)

    log "Gefunden: ${#hevc_videos[@]} HEVC-Videos, $skipped bereits H.264"

    if [ ${#hevc_videos[@]} -eq 0 ]; then
        log "Keine HEVC-Videos zu konvertieren"
        log "=== Fertig ==="
        return 0
    fi

    # Konvertiere parallel
    printf '%s\n' "${hevc_videos[@]}" | \
        xargs -P "$PARALLEL_JOBS" -I {} bash -c 'convert_video "$@"' _ {}

    # Aufräumen
    rm -rf "$TEMP_DIR"

    log "=== Konvertierung abgeschlossen ==="
    log "HEVC-Videos gefunden: ${#hevc_videos[@]}"
}

# Führe Script aus
main "$@"
