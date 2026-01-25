# Migration: Annotierte Bilder in GUI anzeigen

## Änderungen

### 1. Datenbank-Schema
- **Neue Spalte**: `annotated_image_path` in Tabelle `cam2_recordings`
- Speichert den relativen Pfad zum annotierten Bild (z.B. `annotated/annotated_Camera1_00_20260125115843.jpg`)

### 2. Python (watchdog2.py)
- Neue Methode `update_annotated_image_path()` speichert den Pfad in der DB
- Wird automatisch aufgerufen, wenn ein annotiertes Bild erstellt wird

### 3. GUI (API + JavaScript)
- **Neue API**: `api/get_annotated_image.php` - liefert Pfad zum annotierten Bild für ein Gesicht
- **JavaScript**: `js/app.js` erweitert, zeigt annotiertes Bild in Detail-Modal an

## Installation

### Schritt 1: Datenbank-Migration ausführen

Auf dem Server (z.B. `pve`) ausführen:

```bash
cd ~/python/reolink_AI
php migrate_db.php
```

**Oder alternativ** mit dem bereits vorhandenen `--create-schema` Flag:

```bash
cd ~/python
./run_watchdog.sh --create-schema
```

Die Spalte wird automatisch zur Tabelle hinzugefügt, falls sie noch nicht existiert.

### Schritt 2: Bestehende Daten aktualisieren (optional)

Falls bereits annotierte Bilder vorhanden sind, können diese nachträglich mit dem Recording verknüpft werden:

```bash
php update_existing_annotated.php
```

Dies scannt das `annotated/` Verzeichnis und verknüpft die Bilder mit den entsprechenden Recordings.

### Schritt 3: Testen

1. Öffnen Sie die GUI im Browser: `http://your-server/index.php`
2. Klicken Sie auf ein Gesicht, um die Details anzuzeigen
3. Das annotierte Bild sollte unterhalb der Details angezeigt werden (falls verfügbar)

## Funktionsweise

1. **watchdog2.py** analysiert Bilder/Videos mit `--save-annotated`
2. Für jedes Recording wird **ein** annotiertes Bild erstellt (z.B. `annotated/annotated_Camera1_*.jpg`)
3. Der Pfad wird in `cam2_recordings.annotated_image_path` gespeichert
4. Die GUI lädt für **jedes Gesicht** das annotierte Bild vom zugehörigen Recording (via JOIN)

## Hinweise

- **Ein annotiertes Bild pro Recording**, nicht pro Gesicht
- Gesichter vom selben Recording teilen sich das gleiche annotierte Bild
- Das annotierte Bild zeigt **alle** Detektionen (Personen, Fahrzeuge, bekannte Gesichter mit gelben Boxen)
