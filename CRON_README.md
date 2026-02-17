# 🤖 Reolink AI - Cron Job Setup

## 📁 Skript-Übersicht

### 1️⃣ `run_person.sh` - Person Detection
Führt **person.py** mit GPU-Unterstützung aus:
- ✅ Erkennt Gesichter in Bildern/Videos
- ✅ Speichert Face Embeddings (ArcFace, 512-dim)
- ✅ Erkennt Personen mit YOLO
- ✅ Schreibt alles in die Datenbank

**Parameter:**
```bash
./run_person.sh                    # Standard (nur JPG, Debug)
./run_person.sh --limit 100        # Max 100 Dateien
./run_person.sh --force            # Alles neu analysieren
./run_person.sh --debug --limit 10 # Test-Modus
```

---

### 2️⃣ `run_cluster.sh` - Face Clustering
Führt **cam2_cluster_faces.py** aus:
- ✅ Gruppiert identische Gesichter
- ✅ Verwendet DBSCAN Clustering
- ✅ Basiert auf Cosine Distance der Face Embeddings
- ✅ Schreibt `face_cluster_id` in die DB

**Aufruf:**
```bash
./run_cluster.sh
```

---

### 3️⃣ `run_chain.sh` - Komplette Verarbeitungskette ⭐ **EMPFOHLEN**
Führt beide Schritte nacheinander aus:
1. Person Detection
2. Face Clustering (nur wenn Step 1 erfolgreich)

**Parameter:**
```bash
./run_chain.sh                 # Volle Verarbeitung
./run_chain.sh --limit 500     # Max 500 Dateien
```

---

## ⚙️ Installation (Crontab)

### Schritt 1: Pfade anpassen
Bearbeite `crontab.example` und ersetze `/home/gh/python` mit deinem Pfad:
```bash
nano crontab.example
```

### Schritt 2: Crontab installieren
```bash
crontab -e
```

Füge eine der folgenden Zeilen ein:

#### Option A: Täglich um 2:00 Uhr (Produktiv)
```cron
0 2 * * * /home/gh/python/reolink_AI/run_chain.sh
```

#### Option B: Alle 6 Stunden (mit Limit)
```cron
0 */6 * * * /home/gh/python/reolink_AI/run_chain.sh --limit 1000
```

#### Option C: Stündlich (kleine Batches)
```cron
0 * * * * /home/gh/python/reolink_AI/run_chain.sh --limit 100
```

### Schritt 3: Crontab überprüfen
```bash
crontab -l
```

---

## 📊 Logs

Alle Logs werden automatisch erstellt in:
```
reolink_AI/logs/
├── person.log          # Person Detection (aktuell)
├── person.log.old      # Person Detection (vorheriger Lauf)
├── cluster.log         # Face Clustering (aktuell)
├── cluster.log.old     # Face Clustering (vorheriger Lauf)
├── chain.log           # Komplette Chain (aktuell)
└── chain.log.old       # Komplette Chain (vorheriger Lauf)
```

**Logs werden ÜBERSCHRIEBEN** - kein Müllhaufen mit tausend Log-Dateien!
Das vorherige Log wird als `.old` Backup behalten.

**Logs anschauen:**
```bash
# Aktuelles Chain-Log
cat logs/chain.log

# Vorheriges Chain-Log
cat logs/chain.log.old

# Live-Ansicht (tail -f)
tail -f logs/chain.log

# Nur Fehler anzeigen
grep -i error logs/chain.log
```

---

## 🧪 Test vor Produktiv-Betrieb

**Manueller Test:**
```bash
cd ~/python/reolink_AI

# Test mit 10 Dateien
./run_chain.sh --limit 10

# Prüfe Logs
cat logs/chain_*.log | tail -50
```

**Erwartetes Ergebnis:**
```
✅ Person Detection erfolgreich
✅ Face Clustering erfolgreich
Gesamt-Dauer: 2m 15s
```

---

## 🔧 Troubleshooting

### Problem: "CUDA not available"
**Lösung:** Prüfe CUDA-Installation:
```bash
nvidia-smi
/usr/local/cuda-11.8/bin/nvcc --version
```

### Problem: "Permission denied"
**Lösung:** Skripte ausführbar machen:
```bash
chmod +x run_*.sh
```

### Problem: "ModuleNotFoundError"
**Lösung:** Virtual Environment aktivieren:
```bash
source /home/gh/python/venv_py311/bin/activate
pip install -r requirements.txt
```

### Problem: Keine neuen Gesichter erkannt
**Prüfe:**
1. Sind neue Dateien vorhanden?
   ```bash
   ls -lt /var/www/web1/files/ | head
   ```
2. Sind sie bereits in der DB?
   ```sql
   SELECT COUNT(*) FROM cam2_recordings WHERE DATE(recorded_at) = CURDATE();
   ```
3. Force Re-Processing:
   ```bash
   ./run_person.sh --force --limit 50
   ```

---

## 📈 Performance

| Modus | Dateien/Min | GPU-Auslastung | CPU-Last |
|-------|-------------|----------------|----------|
| JPG-Only | ~600 | 30-50% | Niedrig |
| JPG+MP4 | ~120 | 80-100% | Mittel |
| Force Re-Scan | ~80 | 90-100% | Hoch |

**Empfehlung für Produktiv:**
- Stündlich: `--limit 100` (schnell, kontinuierlich)
- Nächtlich: Keine Limits (vollständig)

---

## 🎯 Best Practices

1. ✅ **Teste erst mit `--limit 10`** bevor du Produktiv gehst
2. ✅ **Verwende `run_chain.sh`** statt separate Skripte
3. ✅ **Log-Rotation einrichten** (verhindert volle Festplatte)
4. ✅ **Monitoring einrichten** (z.B. mit `monit` oder `systemd-timer`)
5. ✅ **Backup der Datenbank** vor großen Re-Processing-Läufen

---

## 📝 Beispiel-Workflow (Produktiv)

```cron
# Täglich um 2:00 Uhr: Volle Verarbeitung
0 2 * * * /home/gh/python/reolink_AI/run_chain.sh

# Stündlich: Neue Dateien (max 200)
0 * * * * /home/gh/python/reolink_AI/run_chain.sh --limit 200
```

**Status prüfen:**
```bash
# Heute verarbeitete Dateien
mysql -u gh -pa12345 wagodb -e "SELECT COUNT(*) FROM cam2_recordings WHERE DATE(recorded_at) = CURDATE();"

# Heute erkannte Gesichter
mysql -u gh -pa12345 wagodb -e "SELECT COUNT(*) FROM cam2_detected_faces WHERE DATE(detected_at) = CURDATE();"

# Cluster-Statistik
./run_cluster.sh
```

---

**Viel Erfolg! 🚀**
