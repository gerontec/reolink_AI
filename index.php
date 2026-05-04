<?php
/**
 * CAM2 Admin - Schnelle Personen-Benennung (5 neueste Gesichter mit hoher Qualität)
 */

require_once __DIR__ . '/config.php';

try {
    $pdo = getDbConnection();
} catch (PDOException $e) {
    die("DB-Fehler: " . htmlspecialchars($e->getMessage()));
}

// --- Statistiken (Neue Tabellenstruktur) ---
$stats = [
    'total_detections'  => $pdo->query("SELECT COUNT(*) FROM cam2_detected_faces")->fetchColumn(),
    'distinct_clusters' => $pdo->query("SELECT COUNT(DISTINCT face_cluster_id) FROM cam2_detected_faces WHERE face_cluster_id IS NOT NULL")->fetchColumn(),
    'named_persons'     => $pdo->query("SELECT COUNT(DISTINCT person_name) FROM cam2_detected_faces WHERE person_name != 'Unknown'")->fetchColumn(),
    'total_recordings'  => $pdo->query("SELECT COUNT(*) FROM cam2_recordings")->fetchColumn(),
];

// Filter
$show_all = isset($_GET['all']);
$date_from = $_GET['date_from'] ?? '2026-05-01';

// --- Ein bestes Gesicht pro Cluster (höchste Konfidenz) ---
// Nur brauchbare Gesichter: face_cluster_id IS NOT NULL (Noise/Outlier ausgeblendet)
$sql = "
    SELECT
        f.id,
        f.confidence,
        f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
        f.person_name,
        f.detected_at,
        f.face_cluster_id,
        r.file_path,
        r.camera_name,
        r.recorded_at,
        r.annotated_image_path,
        (f.bbox_x2 - f.bbox_x1) as width,
        (f.bbox_y2 - f.bbox_y1) as height,
        (SELECT COUNT(*) FROM cam2_detected_faces
         WHERE face_cluster_id = f.face_cluster_id AND detected_at >= ?) as cluster_size
    FROM cam2_detected_faces f
    JOIN cam2_recordings r ON f.recording_id = r.id
    WHERE f.face_cluster_id IS NOT NULL
      AND f.detected_at >= ?
      AND f.id = (
          SELECT id FROM cam2_detected_faces f2
          WHERE f2.face_cluster_id = f.face_cluster_id
            AND f2.detected_at >= ?
          ORDER BY f2.confidence DESC, (f2.bbox_x2 - f2.bbox_x1) DESC
          LIMIT 1
      )
";

if (!$show_all) {
    $sql .= " AND f.person_name = 'Unknown'";
}

$sql .= " ORDER BY f.detected_at DESC";

$stmt = $pdo->prepare($sql);
$stmt->execute([$date_from, $date_from, $date_from]);
$persons = $stmt->fetchAll();

// Vorschläge für Autocomplete (Alle bereits vergebenen Namen)
$named = $pdo->query("
    SELECT person_name, COUNT(*) as count
    FROM cam2_detected_faces
    WHERE person_name != 'Unknown'
    GROUP BY person_name
    ORDER BY person_name
")->fetchAll();
?>
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <title>CAM Admin - Gesichtserkennung</title>
    <link rel="stylesheet" href="css/style.css">
    <style>
        /* Bestehende Styles beibehalten... */
        .person-quick-card img { width: 150px; height: 150px; object-fit: cover; border-radius: 8px; border: 1px solid #ccc; }
        .person-quick-card a img { transition: opacity 0.2s, transform 0.2s; }
        .person-quick-card a:hover img { opacity: 0.85; transform: scale(1.02); }
        .stats .unknown { border-bottom: 4px solid #f44336; }
        .stats .known { border-bottom: 4px solid #4caf50; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>👤 CAM2 Admin - Verschiedene Gesichter (1 pro Cluster)</h1>
            <div style="text-align: center; color: #666; font-size: 0.85em; margin-bottom: 10px;">
                Version 2.2.0 | Deploy: <?= date('d.m.Y H:i') ?> | Branch: claude/add-mp4-confidence-scores-cjF5E
            </div>
            <div class="stats">
                <div class="stat-box">
                    <span class="stat-value"><?= number_format($stats['total_detections']) ?></span>
                    <span class="stat-label">Detektionen Total</span>
                </div>
                <div class="stat-box unknown">
                    <span class="stat-value"><?= number_format($stats['distinct_clusters']) ?></span>
                    <span class="stat-label">Verschiedene Gesichter</span>
                </div>
                <div class="stat-box known">
                    <span class="stat-value"><?= $stats['named_persons'] ?></span>
                    <span class="stat-label">Benannt</span>
                </div>
            </div>
        </header>

        <!-- Link zur YOLO-Detektionen Seite -->
        <div style="text-align: center; margin: 20px 0; padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 8px;">
            <a href="yolo_detections.php" style="color: white; text-decoration: none; font-size: 1.1em; font-weight: bold;">
                🎯 Alle YOLO-Detektionen anzeigen (Personen, Fahrzeuge, Objekte)
            </a>
        </div>

        <div class="filters">
            <form method="GET">
                <label>Ab: <input type="date" name="date_from" value="<?= htmlspecialchars($date_from) ?>"></label>
                <label><input type="checkbox" name="all" <?= $show_all ? 'checked' : '' ?>> Auch benannte</label>
                <button type="submit" class="btn">Aktualisieren</button>
            </form>
        </div>

        <?php if (empty($persons)): ?>
            <div class="no-results">
                <h2>Keine brauchbaren Gesichter gefunden.</h2>
                <p>Es werden nur Cluster (≥2 Erkennungen) angezeigt. Noise/Outlier sind ausgeblendet.</p>
                <p>→ Führe <code>python3 cam2_cluster_faces.py</code> aus, um Cluster zu erstellen.</p>
            </div>
        <?php else: ?>
            <div class="quick-rename">
                <?php foreach ($persons as $person): ?>
                    <div class="person-quick-card">
                        <?php if (!empty($person['annotated_image_path'])): ?>
                            <a href="/web1/<?= htmlspecialchars($person['annotated_image_path']) ?>" target="_blank" title="Klicken für Vollbild mit allen YOLO Detektionen">
                                <img src="/web1/<?= htmlspecialchars($person['annotated_image_path']) ?>" alt="Annotiertes Bild" style="max-width: 300px; height: auto; cursor: pointer;">
                            </a>
                            <div style="font-size: 0.8em; color: #666; margin-top: 5px;">📸 Mit YOLO Detektionen (Klick = Vollbild)</div>
                        <?php else: ?>
                            <img src="api/crop_detection.php?v=2&id=<?= $person['id'] ?>&type=face&size=150" alt="Gesicht">
                            <div style="font-size: 0.8em; color: #999; margin-top: 5px;">⚠️ Nur Gesichts-Crop</div>
                        <?php endif; ?>

                        <div class="person-quick-info">
                            <h4>Gesicht #<?= $person['id'] ?></h4>
                            <p>
                                <strong>Kamera:</strong> <?= htmlspecialchars($person['camera_name']) ?><br>
                                <strong>Zeit:</strong> <?= $person['recorded_at'] ?><br>
                                <strong>Konfidenz:</strong> <?= round($person['confidence']*100, 1) ?>%<br>
                                <strong>🔍 Cluster:</strong> <?= $person['cluster_size'] ?> Erkennungen (ID: #<?= $person['face_cluster_id'] ?>)
                            </p>
                            <?php if ($person['person_name'] !== 'Unknown'): ?>
                                <p style="color:green">✓ Aktuell: <?= htmlspecialchars($person['person_name']) ?></p>
                            <?php endif; ?>
                        </div>

                        <div class="person-quick-actions">
                            <input type="text" id="name_<?= $person['id'] ?>"
                                   placeholder="Name..."
                                   list="nameSuggestions"
                                   onkeypress="if(event.key==='Enter') renamePerson(<?= $person['id'] ?>)">
                            <button class="btn btn-primary" onclick="renamePerson(<?= $person['id'] ?>)">Speichern</button>
                            <button class="btn btn-danger" onclick="deletePerson(<?= $person['id'] ?>)" style="background-color: #f44336;">🗑️ Löschen</button>
                        </div>
                    </div>
                <?php endforeach; ?>
            </div>
        <?php endif; ?>
    </div>

    <datalist id="nameSuggestions">
        <?php foreach ($named as $n): ?>
            <option value="<?= htmlspecialchars($n['person_name']) ?>">
        <?php endforeach; ?>
    </datalist>

    <script>
    async function renamePerson(faceId) {
        const name = document.getElementById('name_' + faceId).value.trim();
        if (!name) return;

        try {
            const response = await fetch('api/rename_person.php', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ face_id: faceId, person_name: name })
            });
            const result = await response.json();
            if (result.success) location.reload();
            else alert(result.error);
        } catch (e) { alert(e); }
    }

    async function deletePerson(faceId) {
        try {
            const response = await fetch('api/delete_faces.php', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ face_ids: [faceId] })
            });
            const result = await response.json();
            if (result.success) location.reload();
            else alert(result.error);
        } catch (e) { alert(e); }
    }
    </script>
</body>
</html>
