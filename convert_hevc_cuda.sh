#!/bin/bash
# HEVC to H.264 CUDA Converter (NVIDIA + Path Fix)
# Firefox-kompatible Videos mit CUDA-Beschleunigung

set -euo pipefail

# --- CUDA PFAD FIX ---
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda-11.8/bin:${PATH:-}

# --- KONFIGURATION ---
BASE_DIR="${1:-/var/www/web1}"  # Durchsucht REKURSIV alle Unterverzeichnisse!
LOG_FILE="/home/gh/python/logs/video_conversion_cuda.log"
TEMP_DIR="/tmp/video_convert_cuda"
PARALLEL_JOBS=2
MAX_DEPTH=10  # Maximale Verzeichnistiefe

mkdir -p "$TEMP_DIR" "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

export -f log
export LOG_FILE TEMP_DIR LD_LIBRARY_PATH

is_hevc() {
    ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of default=noprint_wrappers=1:nokey=1 "$1" 2>/dev/null | grep -q "hevc"
}
export -f is_hevc

convert_video_cuda() {
    local input="$1"
    local basename=$(basename "$input")
    local temp_output="${TEMP_DIR}/${basename%.mp4}_h264.mp4"

    log "🚀 CUDA-Start: $basename"

    if ffmpeg -y -hwaccel cuda -i "$input" \
        -c:v h264_nvenc -preset fast -pix_fmt yuv420p \
        -movflags +faststart -c:a copy \
        "$temp_output" >> "$LOG_FILE" 2>&1; then

        if ffprobe -v error "$temp_output" >/dev/null 2>&1; then
            local old_size=$(stat -c%s "$input" 2>/dev/null || stat -f%z "$input")
            local new_size=$(stat -c%s "$temp_output" 2>/dev/null || stat -f%z "$temp_output")
            local old_mb=$((old_size / 1024 / 1024))
            local new_mb=$((new_size / 1024 / 1024))

            mv "$temp_output" "$input"
            chmod 664 "$input"
            log "✅ Erfolg: $basename (${old_mb}MB → ${new_mb}MB)"
        else
            log "❌ Validierung fehlgeschlagen: $basename"
            rm -f "$temp_output"
        fi
    else
        log "❌ CUDA-Fehler: $basename"
        rm -f "$temp_output"
    fi
}
export -f convert_video_cuda

# --- MAIN ---
log "=== Start Transcoding mit CUDA 11.8 ==="
log "Basis-Verzeichnis: $BASE_DIR (rekursiv bis Tiefe $MAX_DEPTH)"

if command -v nvidia-smi &> /dev/null; then
    log "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader | tee -a "$LOG_FILE"
else
    log "⚠️  nvidia-smi nicht gefunden"
fi

if [ ! -d "$BASE_DIR" ]; then
    log "❌ FEHLER: Verzeichnis nicht gefunden: $BASE_DIR"
    exit 1
fi

log "Suche nach HEVC-Videos..."
files=$(find "$BASE_DIR" -maxdepth "$MAX_DEPTH" -name "*.mp4" -type f 2>/dev/null)

total_mp4=0
hevc_count=0
h264_count=0
converted=0

# Erste Durchlauf: Zähle Videos
for f in $files; do
    ((total_mp4++))
    if is_hevc "$f"; then
        ((hevc_count++))
    else
        ((h264_count++))
    fi
done

log "Gefunden: $total_mp4 MP4-Dateien ($hevc_count HEVC, $h264_count bereits H.264)"

if [ $hevc_count -eq 0 ]; then
    log "✅ Keine HEVC-Videos zu konvertieren - alle bereits Firefox-kompatibel!"
    exit 0
fi

log "Starte Konvertierung von $hevc_count HEVC-Videos..."

# Zweiter Durchlauf: Konvertiere nur HEVC
for f in $files; do
    if is_hevc "$f"; then
        convert_video_cuda "$f" &
        ((converted++))

        if (( converted % PARALLEL_JOBS == 0 )); then
            wait
        fi
    fi
done

wait
log "=== Fertig: $converted/$hevc_count Videos verarbeitet ==="
rm -rf "$TEMP_DIR"
