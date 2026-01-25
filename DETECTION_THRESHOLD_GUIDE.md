# Face Detection Threshold Guide

## Was ist der Detection Threshold?

Der `--det-thresh` Parameter steuert, wie "sicher" InsightFace sein muss, um ein Gesicht zu erkennen.

- **Niedriger Wert** (z.B. 0.3): Mehr Gesichter werden erkannt, auch bei schwierigen Bedingungen
- **Höherer Wert** (z.B. 0.6): Nur sehr klare Gesichter werden erkannt

## Empfohlene Werte

| Wert | Beschreibung | Wann verwenden? |
|------|--------------|-----------------|
| 0.3  | Sehr sensitiv | Schlechte Lichtverhältnisse, weit entfernte Gesichter |
| 0.4  | **Standard** (empfohlen) | Normale Überwachungskamera-Bedingungen |
| 0.5  | InsightFace Default | Gut beleuchtete Szenen, nahe Gesichter |
| 0.6+ | Konservativ | Nur hochwertige Gesichter, wenig False Positives |

## Verwendung

```bash
# Mit Standard-Wert (0.4)
./run_watchdog.sh

# Sehr sensitiv für schwierige Bedingungen
./run_watchdog.sh --det-thresh 0.3

# Konservativ (nur klare Gesichter)
./run_watchdog.sh --det-thresh 0.6

# Mit anderen Parametern kombinieren
./run_watchdog.sh --det-thresh 0.35 --save-annotated --limit 10
```

## Debugging

Der aktuelle Threshold wird beim Start geloggt:

```
Initialisiere AI-Analyzer (Tesla P4 + InsightFace GPU, det_thresh=0.4)...
✓ InsightFace GPU initialisiert
  Detection Threshold: 0.4
```

## Log-Analyse

Früher im Log sehen Sie:
```
⊘ Gesicht verworfen: Detection Score zu niedrig (0.577 < 0.8)
⊘ Gesicht verworfen: Detection Score zu niedrig (0.742 < 0.8)
```

Mit `--det-thresh 0.4` werden diese Gesichter jetzt erkannt (0.577 und 0.742 > 0.4).

## Optimaler Wert finden

1. **Starte mit 0.4** (Standard)
2. Schaue ins Log: Werden gute Gesichter verworfen?
3. **Senke den Wert** wenn zu viele gute Gesichter fehlen
4. **Erhöhe den Wert** wenn zu viele False Positives

## Beispiel-Session

```bash
# Test mit verschiedenen Werten (nur 5 Bilder)
./run_watchdog.sh --det-thresh 0.3 --limit 5 --debug
# → Schaue ins Log, zähle erkannte Gesichter

./run_watchdog.sh --det-thresh 0.4 --limit 5 --debug
# → Vergleiche Ergebnisse

./run_watchdog.sh --det-thresh 0.5 --limit 5 --debug
# → Wähle besten Wert
```

## Note

- **0% Konfidenz** bedeutet "Unknown" (nicht in /opt/known_faces), NICHT schlechte Qualität!
- Der Threshold betrifft nur die **Detektion** (Erkennung dass überhaupt ein Gesicht da ist)
- Die **Recognition** (Zuordnung zu bekannten Personen) ist ein separater Prozess
