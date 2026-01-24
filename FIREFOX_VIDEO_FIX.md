# Firefox Video-Kompatibilität Fix

## Problem
Firefox kann die konvertierten Videos nicht abspielen.

## Ursachen
1. **Audio-Codec**: Firefox unterstützt nicht alle Audio-Codecs, benötigt AAC
2. **H.264 Profile**: High Profile kann Probleme machen
3. **Pixel-Format**: Muss yuv420p sein
4. **Container**: Muss sauber als MP4 formatiert sein

## Lösung

### Option 1: Neues optimiertes Script verwenden

```bash
# Nutze das neue Script
chmod +x convertcam_firefox.sh
./convertcam_firefox.sh
```

### Option 2: Dein vorhandenes Script anpassen

Ersetze die FFmpeg-Zeile in `convertcam.sh` mit:

```bash
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i "$input_file" \
  -vf "scale_cuda=1920:-2" \
  -c:v h264_nvenc \
  -preset p4 \
  -profile:v main \
  -level 4.1 \
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
  -y "$TARGET_FILE"
```

### Wichtige Änderungen erklärt:

**1. H.264 Profile**
```bash
-profile:v main          # Kompatibel mit allen Browsern
-level 4.1               # Level für Full HD
```

**2. Pixel-Format**
```bash
-pix_fmt yuv420p         # Standard-Format für Web-Videos
```

**3. Audio zu AAC konvertieren**
```bash
-c:a aac                 # AAC Audio-Codec (statt copy)
-b:a 128k                # Audio-Bitrate
-ar 48000                # Sample-Rate 48kHz
-ac 2                    # Stereo
```

**4. Container explizit als MP4**
```bash
-f mp4                   # Format explizit setzen
```

## Test nach Konvertierung

### 1. Video-Info prüfen
```bash
ffprobe -v error -show_format -show_streams /var/www/web1/2026mime/01/video.mp4
```

Sollte zeigen:
- **Video**: h264, yuv420p, level 41
- **Audio**: aac, 48000 Hz, stereo

### 2. Im Browser testen
```html
<!-- test.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Video Test</title>
</head>
<body>
    <video width="800" controls>
        <source src="/2026mime/01/video.mp4" type="video/mp4">
        Ihr Browser unterstützt das Video-Tag nicht.
    </video>
</body>
</html>
```

### 3. MIME-Type prüfen
```bash
# Apache/Nginx muss korrekte MIME-Types setzen
# In .htaccess oder nginx.conf:
AddType video/mp4 .mp4
```

## Troubleshooting

### Video lädt nicht
```bash
# Prüfe moov-Atom Position (sollte am Anfang sein)
ffprobe -v error -show_entries format_tags=major_brand /path/to/video.mp4
```

### Audio fehlt
```bash
# Prüfe Audio-Stream
ffprobe -v error -select_streams a:0 -show_entries stream=codec_name /path/to/video.mp4
```

### Langsames Laden
```bash
# Prüfe ob faststart gesetzt ist
ffmpeg -i video.mp4 2>&1 | grep "moov atom"
# Sollte NICHT "moov atom not at the beginning" zeigen
```

## Performance

**Vorher (mit -c:a copy):**
- Schneller (kein Audio-Encoding)
- Aber: Kompatibilitätsprobleme möglich

**Nachher (mit -c:a aac):**
- Etwas langsamer (~10% mehr Zeit)
- Aber: Funktioniert in allen Browsern

**GPU-Nutzung bleibt gleich** - Video wird weiterhin mit NVENC encodiert.

## Batch-Konvertierung bestehender Videos

Falls du bereits konvertierte Videos hast die nicht funktionieren:

```bash
# Finde alle Videos in *mime Ordnern
find /var/www/web1 -path "*mime/*/*.mp4" -type f | while read f; do
    echo "Rekonvertiere: $f"

    # Temporäre Datei
    tmp="${f}.tmp.mp4"

    # Neu encodieren mit AAC Audio
    ffmpeg -i "$f" \
      -c:v copy \
      -c:a aac -b:a 128k -ar 48000 -ac 2 \
      -movflags +faststart \
      -f mp4 \
      "$tmp" && mv "$tmp" "$f"
done
```

## Zusammenfassung

✅ **Wichtigste Änderungen:**
1. `-c:a aac` statt `-c:a copy`
2. `-profile:v main` für Kompatibilität
3. `-pix_fmt yuv420p` explizit setzen
4. `-f mp4` Container-Format explizit

Diese Einstellungen garantieren, dass die Videos in **allen modernen Browsern** funktionieren:
- ✅ Firefox
- ✅ Chrome
- ✅ Safari
- ✅ Edge
- ✅ Mobile Browser
